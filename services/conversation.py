import json
import os
from openai import OpenAI
from google import genai

from services.redis_session import (
    get_session,
    create_session,
    update_crop_advice_category,
    update_crop_info,
    update_is_existing_crop,
    update_district_info,
    update_user_query,
    delete_session,
    update_session_state,
    set_user_location,
    append_user_query,
    update_session,
    SessionState
)
from services.status import Status
from services.message import Message
from services.graph_api import GraphApi
from services.weather import send_weather
from services.audio import process_voice_note
from services.vision import analyze_image
from services.crop_name import detect_crop
from services.config import Config
from services.utility import set_timeout

_client = OpenAI(api_key=Config.openai_api_key)
_gemini = genai.Client(api_key=Config.gemini_api_key)

SYSTEM_INSTRUCTION = """
You are a Senior Agronomist and Citrus Specialist specialized in Haryana Agricultural University (HAU) recommendations. 

TASK: Provide an exhaustive list of all distinct varieties, clonal selections, and improved strains for the requested crop suited for Haryana.

STEP-BY-STEP LOGIC:
1. GRANULAR SELECTION: Do not treat the crop name as the only variety. You must identify specific selections (e.g., for Kinnow, look for 'Seedless Kinnow', 'PAU Kinnow 1', 'Kinnow-82', or related commercial hybrids like 'Daisy').
2. HARYANA SUITABILITY: Only include varieties officially released or recommended by HAU Hisar, ICAR-NRCC, or PAU Ludhiana for the North Indian plains.
3. DATA POINTS: For each entry, provide:
   - Sowing/Planting window for Haryana.
   - A detailed Hindi description with **bold** keywords.
   - Specific yield data (e.g., "500-800 fruits per tree" or "20-25 tonnes per hectare").

STRICT JSON OUTPUT FORMAT:
{
  "crop_name": "[Input Crop]",
  "varieties": [
    {
      "variety_name": "[Full Technical Name]",
      "sowing_time": "[Specific Months]",
      "description": "[Detailed Hindi Description with bolded keywords and yield stats]"
    }
  ]
}

STRICT RULES:
- YOU MUST PROVIDE A LIST OF MULTIPLE DISTINCT ENTRIES. ONE ENTRY IS NOT ACCEPTABLE.
- DO NOT return markdown blocks (NO ```json).
- THE RESPONSE MUST START WITH { AND END WITH }.
- Focus on North Indian conditions (heat tolerance/frost resistance).
"""


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
            GraphApi.message_text(
                        sender_phone_number_id,
                        message.from_,
                        "नमस्कार किसान भाई/बहन, आपका स्वागत है। यहाँ आप फसल और मौसम से जुड़े सवाल पूछ सकते हैं।"
                    )
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
                        "मौसम जानने के लिए अपना लोकेशन भेजें।"
                    )
                else:
                    update_crop_advice_category(message.from_, category_id)
                    update_session_state(message.from_, SessionState["AWAITING_DISTRICT_NAME"])
                    GraphApi.message_text(
                        sender_phone_number_id,
                        message.from_,
                        "कृपया अपने ज़िले का नाम बताइए?"
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
                set_timeout(2,
                            GraphApi.send_welcome_menu,
                            message.id,
                            sender_phone_number_id,
                            message.from_
                            )
            else:
                _reset_session_state(message.id, sender_phone_number_id, message.from_)
            return

        if state == SessionState["AWAITING_DISTRICT_NAME"]:
            if message.type != "text":
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया अपने ज़िले का नाम बताइए।"
                )
                return
            update_district_info(message.from_, message.text)
            update_session_state(message.from_, SessionState["AWAITING_CROP_NAME"])
            GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया फसल का नाम टाइप करें।"
                )
            return
            
        if state == SessionState["AWAITING_CROP_NAME"]:
            if message.type != "text":
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया फसल का नाम टाइप करें।"
                )
                return

            detect = detect_crop(message.text)  # NEW: dict response

            # Case: ambiguous (curated OR fuzzy)
            if detect.get("is_ambiguous"):
                # Persist something so we can resolve after click
                # store the options in session so we can map click -> crop
                update_session(message.from_, {
                    "ambiguousCropOptions": detect.get("ambiguous_crop_names", []),
                    "ambiguousButtonOptions": detect.get("button_options", []),
                })
                update_session_state(message.from_, SessionState["AWAITING_AMBIGUOUS_CROP_CHOICE"])

                # Build button titles:
                # Prefer curated Hindi labels if provided; else use crop names.
                button_options = detect.get("button_options") or []
                if button_options:
                    # Convert list of strings to button dicts
                    buttons = [
                        {"id": f"amb_crop_{i}", "title": t[:20]}  # WA title limit ~20 chars
                        for i, t in enumerate(button_options[:3])
                    ]
                    title = "आप किस फसल के बारे में पूछ रहे हैं? कृपया चुनें:"
                else:
                    names = detect.get("ambiguous_crop_names", [])[:3]
                    buttons = [
                        {"id": f"amb_crop_{i}", "title": name[:20]}
                        for i, name in enumerate(names)
                    ]
                    title = "फसल का नाम स्पष्ट नहीं है। कृपया सही फसल चुनें:"

                GraphApi.send_ambiguous_crop_menu(
                    message.id,
                    sender_phone_number_id,
                    message.from_,
                    title,
                    buttons
                )
                return

            # Case: none
            crop = detect.get("crop_name")
            if not crop:
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "मुझे फसल का नाम पहचान नहीं आया। कृपया फसल का नाम फिर से बताएं।"
                )
                return

            # Case: found a single crop -> ask for confirmation
            matched_by = detect.get("matched_by")
            is_existing = detect.get("is_existing_crop", matched_by == "local")
            crop_hi = detect.get("crop_name_hi") or crop

            update_session(message.from_, {
                "pendingCrop": crop,
                "pendingCropHi": crop_hi,
                "pendingIsExistingCrop": is_existing,
                "pendingMatchedBy": matched_by,
            })
            update_session_state(message.from_, SessionState["AWAITING_CROP_CONFIRMATION"])
            GraphApi.send_crop_confirmation_menu(
                message.id,
                sender_phone_number_id,
                message.from_,
                crop_hi
            )
            return

        if state == SessionState["AWAITING_CROP_CONFIRMATION"]:
            if not (interaction and interaction.get("kind") in ("BUTTON", "REPLY")):
                session = get_session(message.from_) or create_session(message.from_)
                crop_hi = session.get("pendingCropHi")
                if crop_hi:
                    GraphApi.send_crop_confirmation_menu(
                        message.id,
                        sender_phone_number_id,
                        message.from_,
                        crop_hi
                    )
                else:
                    update_session_state(message.from_, SessionState["AWAITING_CROP_NAME"])
                    GraphApi.message_text(
                        sender_phone_number_id,
                        message.from_,
                        "कृपया फसल का नाम बताइए।"
                    )
                return

            reply_id = interaction.get("id")
            if reply_id == "crop_confirm_no":
                update_session(message.from_, {
                    "pendingCrop": None,
                    "pendingCropHi": None,
                    "pendingIsExistingCrop": None,
                    "pendingMatchedBy": None,
                })
                update_session_state(message.from_, SessionState["AWAITING_CROP_NAME"])
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "ठीक है, कृपया फसल का नाम फिर से बताइए।"
                )
                return

            if reply_id == "crop_confirm_yes":
                session = get_session(message.from_) or create_session(message.from_)
                crop = session.get("pendingCrop")
                is_existing = session.get("pendingIsExistingCrop")
                if not crop:
                    update_session_state(message.from_, SessionState["AWAITING_CROP_NAME"])
                    GraphApi.message_text(
                        sender_phone_number_id,
                        message.from_,
                        "कृपया फसल का नाम बताइए।"
                    )
                    return
                update_session(message.from_, {
                    "pendingCrop": None,
                    "pendingCropHi": None,
                    "pendingIsExistingCrop": None,
                    "pendingMatchedBy": None,
                })
                _continue_after_crop_selected(
                    message.id,
                    sender_phone_number_id,
                    message.from_,
                    crop,
                    bool(is_existing)
                )
                return

            GraphApi.message_text(
                sender_phone_number_id,
                message.from_,
                "कृपया हाँ या नहीं चुनें।"
            )
            return

        if state == SessionState["AWAITING_AMBIGUOUS_CROP_CHOICE"]:
            if not (interaction and interaction.get("kind") in ("BUTTON", "REPLY")):
                # user didn't click a button
                GraphApi.message_text(
                    sender_phone_number_id,
                    message.from_,
                    "कृपया नीचे दिए गए विकल्पों में से एक चुनें।"
                )
                return

            # Button id will be like "amb_crop_0"
            picked_id = interaction.get("id")  # e.g. amb_crop_1
            try:
                idx = int(picked_id.split("_")[-1])
            except Exception:
                GraphApi.message_text(sender_phone_number_id, message.from_, "कृपया विकल्प फिर से चुनें।")
                return

            session = get_session(message.from_) or create_session(message.from_)
            options = session.get("ambiguousCropOptions", []) or []
            if idx < 0 or idx >= len(options):
                GraphApi.message_text(sender_phone_number_id, message.from_, "कृपया विकल्प फिर से चुनें।")
                return

            crop = options[idx]
            _continue_after_crop_selected(
                message.id,
                sender_phone_number_id,
                message.from_,
                crop,
                True
            )
            return

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


def _continue_after_crop_selected(message_id, sender_phone_number_id, user_id, crop, is_existing):
    update_crop_info(user_id, crop)
    update_is_existing_crop(user_id, bool(is_existing))

    session = get_session(user_id) or create_session(user_id)
    category_id = session.get("cropAdviceCategory")
    if category_id == "variety_sowing_time":
        response = _get_varieties_sowing_response(crop)
        if not response:
            GraphApi.message_text(
                sender_phone_number_id,
                user_id,
                "माफ़ कीजिए, इस फसल के लिए किस्में और बुवाई का समय उपलब्ध नहीं है।"
            )
        else:
            GraphApi.message_text(sender_phone_number_id, user_id, response)

        update_session_state(user_id, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
        set_timeout(
            2,
            GraphApi.send_welcome_menu,
            message_id,
            sender_phone_number_id,
            user_id
        )
        return

    if category_id in ("variety", "sowing_time"):
        response = _format_varieties_response(crop)
        if not response:
            GraphApi.message_text(
                sender_phone_number_id,
                user_id,
                "माफ़ कीजिए, इस फसल के लिए किस्में उपलब्ध नहीं हैं। कृपया दूसरी फसल का नाम बताइए।"
            )
            return

        GraphApi.message_text(sender_phone_number_id, user_id, response)
        update_session_state(user_id, SessionState["GREETING"])
        return

    update_session_state(user_id, SessionState["CROP_ADVICE_QUERY_COLLECTING"])
    GraphApi.message_text(
        sender_phone_number_id,
        user_id,
        "कृपया फसल से जुड़ी अपनी समस्या या सवाल बताइए।"
    )

def _reset_session_state(message_id, phone_number_id, user_id):
    update_session_state(user_id, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
    GraphApi.send_welcome_menu(message_id, phone_number_id, user_id)


def _generate_response(session):
    if not session:
        return "Session expired. Please try again."

    try:
        user_query_text = " ".join(session.get("query", {}).get("texts", []))
        det = detect_crop(user_query_text)

        if det.get("is_ambiguous"):
            opts = det.get("ambiguous_crop_names", [])[:3]
            return "फसल का नाम स्पष्ट नहीं है। कृपया इनमें से सही फसल का नाम बताएं: " + ", ".join(opts)

        crop = det.get("crop_name")
        if not crop:
            return "मुझे फसल का नाम पहचान नहीं आया। कृपया फसल का नाम फिर से बताएं।"

        update_crop_info(session["userId"], crop)
        update_is_existing_crop(
            session["userId"],
            det.get("is_existing_crop", det.get("matched_by") == "local")
        )


        category_id = session.get("cropAdviceCategory")
        if category_id == "variety_sowing_time":
            response = _get_varieties_sowing_response(crop)
            return response or "माफ़ कीजिए, इस फसल के लिए किस्में और बुवाई का समय उपलब्ध नहीं है।"

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


def _get_varieties_sowing_response(crop):
    records = _load_varieties_records(crop)
    if records:
        return _format_varieties_sowing_response(crop, records)

    return _fetch_varieties_from_gemini(crop)


def _format_varieties_sowing_response(crop, records):
    lines = [f"{crop} की किस्में और बुवाई का समय:"]
    for record in records:
        variety = record.get("Variety") or "N/A"
        sowing_time = record.get("Sowing_Time") or record.get("Sowing Time") or "N/A"
        lines.append(f"- {variety} — {sowing_time}")
    return "\n".join(lines)


def _fetch_varieties_from_gemini(crop):
    prompt = f"{SYSTEM_INSTRUCTION}\n\nCrop: {crop}"
    try:
        response = _gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={"temperature": 0}
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        print(f"Gemini error: {exc}")
        return None

    json_text = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    return _format_gemini_varieties_json(parsed)


def _format_gemini_varieties_json(parsed):
    if not isinstance(parsed, dict):
        return None

    crop_name = parsed.get("crop_name") or ""
    varieties = parsed.get("varieties")
    if not isinstance(varieties, list) or not varieties:
        return None

    lines = [f"{crop_name} की किस्में और बुवाई का समय:" if crop_name else "किस्में और बुवाई का समय:"]
    for entry in varieties:
        if not isinstance(entry, dict):
            continue
        variety_name = entry.get("variety_name") or "N/A"
        sowing_time = entry.get("sowing_time") or "N/A"
        description = entry.get("description") or ""
        lines.append(f"- {variety_name} — {sowing_time}")
        if description:
            lines.append(description)
    return "\n".join(lines) if len(lines) > 1 else None


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
