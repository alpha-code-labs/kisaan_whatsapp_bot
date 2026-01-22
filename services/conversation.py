import json
import os
import uuid
import logging
from openai import OpenAI
from google import genai
import mimetypes
import time

from services.redis_session import (
    get_session,
    create_session,
    update_crop_advice_category,
    update_crop_info,
    update_is_existing_crop,
    update_district_info,
    update_user_query,
    update_session_state,
    set_user_location,
    append_user_query,
    reset_query_arrays,
    append_advice_response,
    next_upload_count,
    update_session,
    dump_session,
    delete_session,
    append_aggregated_query_response,
    append_aggregated_query_decomposed_response,
    SessionState
)
from services.status import Status
from services.message import Message
from services.graph_api import GraphApi
from services.weather import send_weather
from services.crop_name import detect_crop
from services.blob_storage import BlobStorageService
from services.config import Config
from services.rag_builder import retrieve_rag_evidence
from services.utility import set_timeout

_client = OpenAI(api_key=Config.openai_api_key)
_gemini = genai.Client(api_key=Config.gemini_api_key)

_blob_storage = None
logger = logging.getLogger("conversation")

def get_blob_storage():
    global _blob_storage
    if _blob_storage is None:
        _blob_storage = BlobStorageService()
    return _blob_storage

# _blob_storage = BlobStorageService()

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

AUDIT_SYSTEM_INSTRUCTION = """
You are a Senior Agricultural Scientist at Haryana Agricultural University (HAU) Hisar. 
Your task is to audit and fact-check a JSON object containing crop varieties and sowing times.

TASK:
1. VALIDATE: Check if each variety name actually exists and is recommended for Haryana.
2. REMOVE HALLUCINATIONS: If a variety is made up or purely tropical, remove it.
3. CORRECT SOWING TIMES: Ensure the 'sowing_time' aligns with Haryana's Rabi/Kharif seasons.
4. REWRITE DESCRIPTIONS: Refine Hindi terminology. Bold **पैदावार**, **बीमारी**, and **मौसम**.

STRICT OUTPUT RULES:
- Return ONLY the corrected JSON object.
- NO markdown blocks (NO ```json).
- NO conversational filler.
"""

MULTIMODAL_SYSTEM_INSTRUCTION = """
You are an expert Agricultural Communication Specialist. Your task is to aggregate Text, Audio, and Image inputs into a single English question focused strictly on a LOCKED CROP.

LOGIC:
1. ABSOLUTE CROP VALIDATION (PRIORITY 1): 
   - You are provided with a 'Locked Crop Name'.
   - MANDATORY STEP: Before anything else, scan EVERY text line, EVERY audio transcript, and EVERY image.
   - If ANY input explicitly mentions or depicts a DIFFERENT crop (e.g., you see a weed on Sugarcane but the locked crop is Wheat), you MUST immediately return ONLY: "This is not a question about {Locked Crop Name}".
   - DO NOT combine, synthesize, or mention the other crop in a question. If there is a mismatch, REJECT the entire query.

2. MULTIMODAL AUDIT & CENSUS: 
   - Once validation passes, conduct a comprehensive census.
   - IMAGES: Analyze EVERY image. Identify all pests, diseases, and WEEDS visible.
   - TEXT/AUDIO: Read every line (including translated Hindi) and all audio notes.
   - Ensure distinct issues from different images (e.g., Aphids in one, Weeds in another) are both captured.

3. PARALLEL ITEMIZATION: Combine unique concerns into a compound English sentence using "and".

4. FORMAT: Exactly: {Locked Crop Name} - [Question 1] and [Question 2] and [Question 3]?

STRICT RULES:
- NO introductory text.
- NO markdown blocks.
- If a different crop appears in even one of many files, trigger the rejection message.
"""

MULTIMODAL_DECOMPOSITION_SYSTEM_INSTRUCTION = """
You are a Query Decomposition Expert for an Agricultural RAG system. 
Your task is to take a compound query and break it into a list of individual, atomic technical questions.

VALIDATION RULES:
1. ATOMICITY: Each line must address exactly ONE technical issue.
2. CROP LOCKING: Every line MUST start with the crop name followed by a pipe symbol '|'.
3. SEARCH OPTIMIZATION: Include technical keywords like 'dosage', 'control', or 'timing'.
4. NO FORMATTING: Output only the list, one per line. No bullets or numbers.

INPUT FORMAT: "Wheat - Fertilizer and Thrips..."
OUTPUT FORMAT: 
Wheat | What are the recommended fertilizer types for wheat?
Wheat | How to control thrips in wheat crops?
"""

AGRI_ADVICE_SYSTEM_INSTRUCTION = """
You are a highly experienced Senior Agricultural Scientist. Your task is to provide strictly factual, technical, and non-hallucinated agronomic advice in Hindi.

INPUT FORMAT: 
You will receive a compound question in the format: "{CropName} - [Concern 1] and [Concern 2] and [Concern 3]?"

LOGIC:
1. DECONSTRUCTION: Break down the "and"-separated compound question into its individual technical components (e.g., Fertilizer, Irrigation, Pests, Growth Issues, Weeds).
2. FACTUAL RESPONSE: Provide accurate advice based on established agricultural science for the specific crop mentioned. If a question is outside the scope of factual agronomy, state that clearly.
3. LANGUAGE POLICY: The entire response must be in Hindi script. No English words or characters should be used in the final output.
4. TONE: Professional, helpful, and expert.

STRICT RESPONSE FORMAT (HINDI ONLY):
- Opening: "किसान भाई, यह रहा आपके सवालों का उत्तर।"
- Body: Each technical topic must have its own numbered header in Hindi, followed by the specific advice.
- Example:
  1) [Hindi Topic Header]
  [Detailed Hindi advice]
  
  2) [Hindi Topic Header]
  [Detailed Hindi advice]

STRICT RULES:
- NO introductory English filler.
- NO markdown code blocks.
- ZERO hallucinations. 
"""


AGRI_ADVICE_AUDIT_SYSTEM_INSTRUCTION = """
You are a Senior Agricultural Auditor and Fact-Checker. Your task is to review an existing Hindi agronomic response for absolute scientific accuracy and safety.

LOGIC:
1. SCIENTIFIC VERIFICATION: Review every technical claim made in the Hindi text (Fertilizer doses, Chemical names, Irrigation timings, etc.).
2. SAFETY CHECK: Ensure no toxic or incompatible chemicals are recommended together and that dosages are safe for the specific crop.
3. CORRECTION: If you find an error, you must correct it in the final version. If the information is missing a crucial safety warning, add it.
4. LANGUAGE POLICY: The entire response must remain in Hindi script. No English characters.

STRICT RESPONSE FORMAT (HINDI ONLY):
- Retain the original structure:
  Opening: "किसान भाई, यह रहा आपके सवालों का उत्तर।"
  followed by the numbered headers and detailed advice.

STRICT RULES:
- If the original response is 100% correct, you may keep it as is, but ensure the tone remains expert.
- DO NOT add introductory filler like "I have audited this." 
- Output ONLY the final, corrected Hindi response.
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
                #dump session, create a fresh one
                dump_session(message.from_)
                delete_session(message.from_)
                create_session(message.from_)
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
                get_blob_storage()
                audio_buffer = GraphApi.download_audio(message.audio["id"])
                session = get_session(message.from_) or create_session(message.from_)
                session_id = _ensure_session_id(message.from_, session)
                count = next_upload_count(message.from_)
                mime_type = message.audio.get("mimeType")
                ext = BlobStorageService.guess_extension(mime_type)
                blob_name = f"{message.from_}_{session_id}_{count}{ext}"
                url = _blob_storage.upload_bytes(blob_name, audio_buffer, mime_type)
                append_user_query(message.from_, {"audioUrl": url})

            if message.type == "image":
                get_blob_storage()
                image_buffer = GraphApi.download_image(message.image["id"])
                session = get_session(message.from_) or create_session(message.from_)
                session_id = _ensure_session_id(message.from_, session)
                count = next_upload_count(message.from_)
                mime_type = message.image.get("mimeType")
                ext = BlobStorageService.guess_extension(mime_type)
                blob_name = f"{message.from_}_{session_id}_{count}{ext}"
                url = _blob_storage.upload_bytes(blob_name, image_buffer, mime_type)
                append_user_query(message.from_, {"imageUrl": url})

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
                    "कृपया प्रतीक्षा करें, हम आपके अनुरोध पर कार्य कर रहे हैं।"
                )
                response = _generate_response(get_session(message.from_))

                #dump session
                dump_session(message.from_)
                delete_session(message.from_)
                create_session(message.from_)
                if isinstance(response, dict):
                    GraphApi.message_text(sender_phone_number_id, message.from_, response.get("text", ""))
                else:
                    GraphApi.message_text(sender_phone_number_id, message.from_, response)
                    # reset_query_arrays(message.from_)
                    update_session_state(message.from_, SessionState["GREETING"])

            return

        if state == SessionState["PROCESSING_CROP_QUERY"]:
            response = _generate_response(get_session(message.from_))
            dump_session(message.from_)
            delete_session(message.from_)
            create_session(message.from_)
            if isinstance(response, dict):
                GraphApi.message_text(sender_phone_number_id, message.from_, response.get("text", ""))
            else:
                GraphApi.message_text(sender_phone_number_id, message.from_, response)
                # reset_query_arrays(message.from_)
                update_session_state(message.from_, SessionState["GREETING"])
            return

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
        
        #dump session
        dump_session(user_id)
        delete_session(user_id)
        create_session(user_id)
        update_session_state(user_id, SessionState["AWAITING_MENU_WEATHER_CHOICE"])
        set_timeout(
            2,
            GraphApi.send_welcome_menu,
            message_id,
            sender_phone_number_id,
            user_id
        )
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
        print("starting response generation")
        start = time.perf_counter()
        query = session.get("query", {}) or {}
        texts = query.get("texts", []) or []
        audio_urls = query.get("audios", []) or []
        image_urls = query.get("images", []) or []
        user_query_text = " ".join(texts)

        crop = session.get("crop")
        if not crop:
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
            print(f"starting aggregation query")
            aggregated = _aggregate_multimodal_query(crop, texts, audio_urls, image_urls)
            print(f"completed aggregation query")
            if aggregated.get("status") == "mismatch":
                reset_query_arrays(session["userId"])
                # the quesion is not about the crop, we should dump the failed session as well before resetting
                dump_session(session["userId"], True)
                update_session_state(session["userId"], SessionState["CROP_ADVICE_QUERY_COLLECTING"])
                return {
                    "text": f"कृपया {crop} के बारे में ही पूछें।",
                    "action": "reset_query",
                }
            if aggregated.get("status") != "ok":
                return "कृपया अधिक विवरण जोड़ें।"
            
            #identify if the question is about a new crop
            aggregated_query = aggregated.get("text", "").strip()
            append_aggregated_query_response(session["userId"], aggregated_query)
            print(aggregated_query)

            if not aggregated_query:
                return None

            if not session["isExistingCrop"]:
                try:
                    # -------- First call: main advice --------
                    prompt_main = f"""
            {AGRI_ADVICE_SYSTEM_INSTRUCTION}

            User query:
            {aggregated_query}
            """.strip()

                    response_main = _gemini.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt_main
                    )

                    raw_response = response_main.text.strip()

                    # -------- Second call: audit / refinement --------
                    prompt_audit = f"""
            {AGRI_ADVICE_AUDIT_SYSTEM_INSTRUCTION}

            Previous answer:
            {raw_response}
            """.strip()

                    response_audit = _gemini.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt_audit
                    )

                    final_response = response_audit.text.strip()

                    append_advice_response(session["userId"], final_response)
                    return final_response

                except Exception as e:
                    logger.exception("Gemini error in crop advice generation")
                    return "कुछ तकनीकी समस्या आ गई है। कृपया थोड़ी देर बाद पुनः प्रयास करें।"
            else:
                response_text = aggregated_query
                print("starting decomposition prompt")
                append_advice_response(session["userId"], response_text)
                print("completed decomposition prompt")

                # --- Second call: Decompose into atomic RAG questions ---
                try:
                    decomposition_prompt = f"""
            {MULTIMODAL_DECOMPOSITION_SYSTEM_INSTRUCTION}

            INPUT:
            {aggregated_query}
            """.strip()

                    decomp_resp = _gemini.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=decomposition_prompt
                    )

                    decomp_text = (decomp_resp.text or "").strip()

                    # Parse into array: one non-empty line per query
                    decomposed_queries = [
                        line.strip()
                        for line in decomp_text.splitlines()
                        if line.strip()
                    ]

                    # (Optional but recommended) basic validation: must contain '|'
                    decomposed_queries = [
                        q for q in decomposed_queries
                        if "|" in q
                    ]

                    # Store in an array (session + your existing persistence if you have it)
                    # session["decomposedQueries"] = decomposed_queries
                    append_aggregated_query_decomposed_response(session["userId"], decomposed_queries)

                    try:
                        rag_results = retrieve_rag_evidence(decomposed_queries)
                        update_session(session["userId"], {"ragResults": rag_results})
                    except Exception:
                        logger.exception("RAG retrieval error")

                    #dump the session at this point for now
                    dump_session(session["userId"])
                    delete_session(session["userId"])
                except Exception:
                    logger.exception("Gemini error in query decomposition")
                    session["decomposedQueries"] = []

                return response_text
                # this will go to the RAG corpus


            
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

    audited = _audit_varieties_with_gemini(parsed)
    if audited:
        parsed = audited

    return _format_gemini_varieties_json(parsed)


def _audit_varieties_with_gemini(parsed):
    try:
        payload = json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return None

    prompt = f"{AUDIT_SYSTEM_INSTRUCTION}\n\nJSON:\n{payload}"
    try:
        response = _gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={"temperature": 0}
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        print(f"Gemini audit error: {exc}")
        return None

    json_text = raw.replace("```json", "").replace("```", "").strip()
    try:
        audited = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    return audited


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


def _aggregate_multimodal_query(crop, texts, audio_urls, image_urls):
    """
    Build a true multimodal Gemini request using PUBLIC URLs as file parts
    (NOT as plain text inside JSON).
    """
    if not crop:
        return {"status": "error"}

    texts = texts or []
    audio_urls = audio_urls or []
    image_urls = image_urls or []

    if not (texts or audio_urls or image_urls):
        return {"status": "error"}

    # Lazy import so you don't have to change the top of your file.
    try:
        from google.genai import types
    except Exception as exc:
        logger.exception("Failed to import google.genai.types: %s", exc)
        return {"status": "error"}

    def _guess_mime(url: str, kind: str) -> str:
        """
        Best-effort MIME guess from URL extension.
        Falls back to sane defaults if unknown.
        """
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            return guessed

        if kind == "image":
            return "image/jpeg"
        if kind == "audio":
            return "audio/ogg"
        return "application/octet-stream"

    # Inject locked crop into the instruction text (your template uses {Locked Crop Name})
    instruction = MULTIMODAL_SYSTEM_INSTRUCTION.replace("{Locked Crop Name}", crop).strip()

    # Put your instruction + any farmer text as text parts.
    # (Gemini will also see the media parts we attach below.)
    text_blob = "\n".join([t for t in texts if isinstance(t, str) and t.strip()]).strip()
    user_text_part = f"{instruction}\n\nLOCKED_CROP_NAME: {crop}\n"
    if text_blob:
        user_text_part += f"\nFARMER_TEXT:\n{text_blob}\n"

    parts = [types.Part.from_text(text=user_text_part)]

    # Attach images as actual file parts (Gemini will fetch and analyze them)
    for url in image_urls:
        if not isinstance(url, str) or not url.strip():
            continue
        mime_type = _guess_mime(url, "image")
        parts.append(types.Part.from_uri(file_uri=url, mime_type=mime_type))

    # Attach audios as actual file parts
    for url in audio_urls:
        if not isinstance(url, str) or not url.strip():
            continue
        mime_type = _guess_mime(url, "audio")
        parts.append(types.Part.from_uri(file_uri=url, mime_type=mime_type))

    try:
        response = _gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Content(role="user", parts=parts)],
            config={"temperature": 0},
        )
        raw = (response.text or "").strip()
        logger.info("[gemini][query_aggregation] raw=%r", raw)
    except Exception as exc:
        logger.exception("Gemini aggregation error: %s", exc)
        return {"status": "error"}

    expected = f"This is not a question about {crop}"
    if raw.strip().lower() == expected.lower():
        return {"status": "mismatch"}

    if not raw:
        return {"status": "error"}

    return {"status": "ok", "text": raw}


def _ensure_session_id(user_id, session):
    session_id = session.get("sessionId")
    if not session_id:
        session_id = uuid.uuid4().hex[:8]
        update_session(user_id, {"sessionId": session_id})
    return session_id

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
