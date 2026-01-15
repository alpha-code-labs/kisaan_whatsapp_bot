import json
import os
from openai import OpenAI

from services.redis_session import (
    get_session,
    create_session,
    update_crop_advice_category,
    update_crop_info,
    update_user_query,
    delete_session,
    update_session_state,
    set_user_location,
    append_user_query,
    SessionState
)
from services.status import Status
from services.message import Message
from services.graph_api import GraphApi
from services.weather import send_weather
from services.audio import process_voice_note
from services.vision import analyze_image
from services.crop_detection import detect_crop
from services.config import Config

_client = OpenAI(api_key=Config.openai_api_key)


class Conversation:
    def __init__(self, phone_number_id):
        self.phone_number_id = phone_number_id

    @staticmethod
    def handle_message(sender_phone_number_id, raw_message):
        message = Message(raw_message)
        interaction = message.get_interaction()

        session = get_session(message.from_)
        if not session:
            session = create_session(message.from_)

        state = session.get("state")

        if state == SessionState["GREETING"]:
            GraphApi.send_welcome_menu(message.id, sender_phone_number_id, message.from_)
            update_session_state(message.from_, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
            return

        if state == SessionState["AWAITING_MENU_WEATHER_CHOICE"]:
            if interaction and interaction.get("kind") == "LIST":
                category_id = interaction.get("id")
                if category_id == "weather_info":
                    update_session_state(message.from_, SessionState["AWAITING_WEATHER_LOCATION"])
                    GraphApi.request_location(
                        sender_phone_number_id,
                        message.from_,
                        "Please share your location to get the weather."
                    )
                else:
                    update_crop_advice_category(message.from_, category_id)
                    update_session_state(message.from_, SessionState["AWAITING_CROP_NAME"])
                    GraphApi.message_text(
                        sender_phone_number_id,
                        message.from_,
                        "Which crop do you want to know about?"
                    )
            else:
                _reset_session_state(message.id, sender_phone_number_id, message.from_)
            return

        if state == SessionState["AWAITING_WEATHER_LOCATION"]:
            if message.type == "location":
                location = message.location
                set_user_location(message.from_, location)
                send_weather(sender_phone_number_id, message.from_, location)
                update_session_state(message.from_, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
                GraphApi.send_welcome_menu(message.id, sender_phone_number_id, message.from_)
            else:
                _reset_session_state(message.id, sender_phone_number_id, message.from_)
            return

        if state == SessionState["AWAITING_CROP_NAME"]:
            if message.type != "text":
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "Please type the crop name."
                )
                return

            crop = detect_crop(message.text)
            if not crop or crop == "none":
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "मैं फसल का नाम पहचान नहीं पाया। कृपया फसल का नाम दोबारा बताएं।"
                )
                return

            update_crop_info(message.from_, crop)
            response = _format_varieties_response(crop)
            if not response:
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "मैं उस फसल के बारे में जानकारी नहीं पा सका। कृपया फसल का नाम दोबारा बताएं।"
                )
                return

            GraphApi.message_text(sender_phone_number_id, message.from_, response)

            category_id = session.get("cropAdviceCategory")
            if category_id in ("variety", "sowing_time"):
                update_session_state(message.from_, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
                GraphApi.send_welcome_menu(message.id, sender_phone_number_id, message.from_)
            else:
                update_session_state(message.from_, SessionState["CROP_ADVICE_QUERY_COLLECTING"])
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया अपनी फसल से संबंधित समस्या बताएं।"
                )
            return

        if state == SessionState["CROP_ADVICE_CATEGORY_MENU"]:
            if interaction and interaction.get("kind") == "LIST":
                category_id = interaction.get("id")
                update_crop_advice_category(message.from_, category_id)
                update_session_state(message.from_, SessionState["CROP_ADVICE_QUERY_COLLECTING"])
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    f"आपने '{interaction.get('title')}' चुना है। कृपया अपनी फसल से संबंधित समस्या बताएं।"
                )
            else:
                _reset_session_state(message.id, sender_phone_number_id, message.from_)
            return

        if state == SessionState["CROP_ADVICE_QUERY_COLLECTING"]:
            if message.type == "text":
                append_user_query(message.from_, {"text": message.text})

            if message.type == "audio":
                audio_buffer = GraphApi.download_audio(message.audio["id"])
                result = process_voice_note(audio_buffer)
                append_user_query(message.from_, {
                    "text": result["hinglish"],
                    "rawAudioTranscript": result["transcript"]
                })

            if message.type == "image":
                image_buffer = GraphApi.download_image(message.image["id"])
                vision_result = analyze_image(image_buffer, message.image.get("mimeType"))
                append_user_query(message.from_, {"text": vision_result.get("tags")})

            if not interaction:
                GraphApi.send_query_confirmation_menu(
                    message.id,
                    sender_phone_number_id,
                    message.from_
                )

            if interaction and interaction.get("id") == "query_continue":
                update_session_state(message.from_, SessionState["CROP_ADVICE_QUERY_COLLECTING"])
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया अधिक विवरण जोड़ें।"
                )

            if interaction and interaction.get("id") == "query_done":
                update_session_state(message.from_, SessionState["PROCESSING_CROP_QUERY"])
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "धन्यवाद। आपके अनुरोध पर प्रक्रिया की जा रही है।"
                )
                response = _generate_response(get_session(message.from_))
                GraphApi.message_text(sender_phone_number_id, message.from_, response)
                delete_session(message.from_)

            return

        if state == SessionState["PROCESSING_CROP_QUERY"]:
            response = _generate_response(get_session(message.from_))
            GraphApi.message_text(sender_phone_number_id, message.from_, response)
            update_session_state(message.from_, SessionState["GREETING"])
            return

        _reset_session_state(message.id, sender_phone_number_id, message.from_)

    @staticmethod
    def handle_status(sender_phone_number_id, raw_status):
        status = Status(raw_status)
        if status.status not in ("delivered", "read"):
            return
        print(
            f"Message {status.message_id} to {status.recipient_phone_number} was {status.status}"
        )


def _reset_session_state(message_id, phone_number_id, user_id):
    update_session_state(user_id, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
    GraphApi.send_welcome_menu(message_id, phone_number_id, user_id)


def _generate_response(session):
    if not session:
        return "Session expired. Please try again."

    try:
        user_query_text = " ".join(session.get("query", {}).get("texts", []))
        crop = detect_crop(user_query_text)
        update_crop_info(session["userId"], crop)

        category_id = session.get("cropAdviceCategory")
        if category_id not in ("variety", "sowing_time"):
            completion = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert agricultural assistant. Provide short, actionable and accurate advice "
                            "to farmers based on their queries in Hindi. For queries that have insecticide or pesticide "
                            "recommendations give specific chemical name and dosage. Answer in the context of Haryana "
                            "state only without using the word Haryana. Format the response for WhatsApp: Use bold for "
                            "section titles only. Do NOT use #, ##, or ###. Use short paragraphs. Use simple language "
                            "suitable for farmers."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Crop: {crop}\nCategory: {session.get('cropAdviceCategory') or 'General'}\n"
                            f"Query: {user_query_text}"
                        )
                    }
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content

        categories_text = _load_varieties_text(crop)
        completion = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert agricultural assistant. Provide short, actionable and accurate advice to farmers "
                        "based on their queries in Hindi. Answer in the context of the given region only. Format the response "
                        "for WhatsApp: Use *bold* for section titles only. Do NOT use #, ##, or ###. Use short paragraphs. "
                        "Use simple language suitable for farmers. The user here wants to know either about the varieties of a "
                        "crop or their sowing time. Use the provided list of varieties and sowing times to answer the query:\n"
                        f"cropVarieties: {categories_text}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Crop: {crop}\nCategory: {session.get('cropAdviceCategory') or 'General'}\n"
                        f"Query: {user_query_text}"
                    )
                }
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content

    except Exception as exc:
        print(f"Error generating response: {exc}")
        return "Sorry, I am having trouble processing that request right now."


def _load_varieties_records(crop):
    path = os.path.join(Config.data_dir, "Varieties and Sowing Time.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    return [
        record for record in records
        if (record.get("Crop") or "").lower() == (crop or "").lower()
    ]


def _format_varieties_response(crop):
    matches = _load_varieties_records(crop)
    if not matches:
        return None

    lines = []
    for record in matches:
        sowing_time = record.get("Sowing_Time") or record.get("Sowing Time") or "N/A"
        lines.append(f"Variety: {record.get('Variety')}, Sowing Time: {sowing_time}")

    return f"{crop} varieties and sowing time:\n" + "\n".join(lines)


def _load_varieties_text(crop):
    matches = _load_varieties_records(crop)

    parts = []
    for record in matches:
        sowing_time = record.get("Sowing_Time") or record.get("Sowing Time") or "N/A"
        parts.append(f"Variety: {record.get('Variety')}, Sowing Time: {sowing_time}")

    return " | ".join(parts)
