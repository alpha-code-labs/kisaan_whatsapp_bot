import requests
from services.config import Config


class GraphApi:
    @staticmethod
    def _make_api_call(message_id, sender_phone_number_id, request_body):
        if message_id:
            typing_body = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
                "typing_indicator": {"type": "text"}
            }
            GraphApi._post(sender_phone_number_id, typing_body)

        return GraphApi._post(sender_phone_number_id, request_body)

    @staticmethod
    def _post(sender_phone_number_id, body):
        url = f"{Config.graph_api_url}/{sender_phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {Config.access_token}"}
        response = requests.post(url, json=body, headers=headers, timeout=30)

        # --- DEBUG: print Meta error response on 4xx/5xx (without changing behavior) ---
        if not response.ok:
            try:
                print(f"[GraphApi] HTTP_ERROR status={response.status_code} url={url}")
                try:
                    print(f"[GraphApi] Response text: {response.text}")
                except Exception:
                    pass
                try:
                    print(f"[GraphApi] Response JSON: {response.json()}")
                except Exception:
                    pass
            except Exception:
                # never let logging cause crashes
                pass
        # ---------------------------------------------------------------------------

        response.raise_for_status()
        return response.json()

    @staticmethod
    def send_welcome_menu(message_id, sender_phone_number_id, recipient_phone_number):
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {
                    "text": "कृपया आगे बढ़ने के लिए एक श्रेणी चुनें।"
                },
                "action": {
                    "button": "Choose Category",
                    "sections": [
                        {
                            "title": "Categories",
                            "rows": [
                                {"id": "weather_info", "title": "मौसम जानकारी"},
                                {"id": "disease_management", "title": "कृषि रोग प्रबंधन"},
                                {"id": "insect_management", "title": "कृषि कीट प्रबंधन"},
                                {"id": "fertilizer_use", "title": "कृषि उर्वरक उपयोग"},
                                {"id": "weed_management", "title": "कृषि खरपतवार नियंत्रण"},
                                {"id": "variety_sowing_time", "title": "कृषि किस्में व बुवाई समय"},
                                {"id": "others", "title": "कृषि अन्य"}
                            ]
                        }
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, body)

    @staticmethod
    def send_query_confirmation_menu(message_id, sender_phone_number_id, recipient_phone_number):
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {
                    "text": "क्या आप और सवाल पूछना चाहते हैं या आपके सवाल पूरे हो गए हैं?"
                },
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": "query_continue", "title": "➕ और जानकारी जोड़ें"}},
                        {"type": "reply", "reply": {"id": "query_done", "title": "✅ जानकारी पूरी हो गई है"}}
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, body)

    @staticmethod
    def request_location(sender_phone_number_id, recipient_phone_number, text):
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "location_request_message",
                "body": {"text": text},
                "action": {"name": "send_location"}
            }
        }
        return GraphApi._make_api_call(None, sender_phone_number_id, body)

    @staticmethod
    def message_text(sender_phone_number_id, recipient_phone_number, text):
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "text",
            "text": {"body": text}
        }
        return GraphApi._make_api_call(None, sender_phone_number_id, body)

    @staticmethod
    def send_crop_advice_category_menu(message_id, sender_phone_number_id, recipient_phone_number):
        request_body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": "कृषि सहायता मेनू"},
                "body": {"text": "कृपया विशेषज्ञ मार्गदर्शन के लिए एक श्रेणी चुनें:"},
                "footer": {"text": "नीचे दी गई किसी श्रेणी पर टैप करें"},
                "action": {
                    "button": "श्रेणियाँ देखें",
                    "sections": [
                        {
                            "title": "उपलब्ध श्रेणियाँ",
                            "rows": [
                                [
                                    {"id": "weather_info", "title": "मौसम जानकारी"},
                                    {"id": "disease_management", "title": "रोग प्रबंधन"},
                                    {"id": "insect_management", "title": "कीट प्रबंधन"},
                                    {"id": "fertilizer_use", "title": "उर्वरक उपयोग"},
                                    {"id": "weed_management", "title": "खरपतवार नियंत्रण"},
                                    {"id": "variety_sowing_time", "title": "किस्में व बुवाई समय"},
                                    {"id": "others", "title": "अन्य"}
                                ]
                            ]
                        }
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, request_body)

    # ----------------------------
    # District picker (interactive list) - PAGINATED (<= 10 rows TOTAL)
    # ----------------------------
    @staticmethod
    def send_district_menu(message_id, sender_phone_number_id, recipient_phone_number, districts, page=0):
        """
        districts: list[str] of district names (22).
        page: 0-based page index.
        WhatsApp constraint (your env): TOTAL rows <= 10 per list message.
        Strategy:
          - show up to 8 districts per page
          - add Back/Next rows as needed
          - IDs:
              dist_0..dist_21 = actual districts
              dist_prev       = previous page
              dist_next       = next page
        """
        districts = districts or []
        per_page = 8  # keeps room for Back+Next => 8 + 2 = 10 rows max

        total = len(districts)
        max_page = (total - 1) // per_page if total > 0 else 0
        try:
            page = int(page)
        except Exception:
            page = 0
        if page < 0:
            page = 0
        if page > max_page:
            page = max_page

        start = page * per_page
        end = min(start + per_page, total)
        page_slice = districts[start:end]

        rows = []
        for i, name in enumerate(page_slice):
            idx = start + i  # global index for stable IDs
            rows.append({
                "id": f"dist_{idx}",
                "title": str(name)[:24]
            })

        # Navigation rows (still LIST rows)
        if page > 0:
            rows.append({"id": "dist_prev", "title": "⬅️ पिछला (Back)"})
        if page < max_page:
            rows.append({"id": "dist_next", "title": "➡️ अगला (Next)"})

        # Safety: never exceed 10
        rows = rows[:10]

        body_text = "कृपया अपना ज़िला चुनें:"
        # Optional small hint, still short
        if max_page > 0:
            body_text = f"कृपया अपना ज़िला चुनें: (पेज {page+1}/{max_page+1})"

        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {"text": body_text},
                "action": {
                    "button": "ज़िला चुनें",
                    "sections": [
                        {
                            "title": "हरियाणा के ज़िले",
                            "rows": rows
                        }
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, body)

    @staticmethod
    def send_ambiguous_crop_menu(message_id, sender_phone_number_id, recipient_phone_number, title_text, options):
        """
        options: list of dicts -> [{"id":"crop_pick_0","title":"1. नींबू"}, ...]
        """
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": title_text},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": opt["id"], "title": opt["title"]}}
                        for opt in options[:3]
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, body)

    @staticmethod
    def send_crop_confirmation_menu(message_id, sender_phone_number_id, recipient_phone_number, crop_name_hi):
        body = {
            "messaging_product": "whatsapp",
            "to": recipient_phone_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": f"क्या आप {crop_name_hi} के बारे में जानना चाहते हैं?"},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": "crop_confirm_yes", "title": "हाँ"}},
                        {"type": "reply", "reply": {"id": "crop_confirm_no", "title": "नहीं"}}
                    ]
                }
            }
        }
        return GraphApi._make_api_call(message_id, sender_phone_number_id, body)

    @staticmethod
    def get_media_url(media_id):
        url = f"{Config.graph_api_url}/{media_id}"
        headers = {"Authorization": f"Bearer {Config.access_token}"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def download_media(url):
        headers = {"Authorization": f"Bearer {Config.access_token}"}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.content

    @staticmethod
    def download_audio(media_id):
        media = GraphApi.get_media_url(media_id)
        return GraphApi.download_media(media.get("url"))

    @staticmethod
    def download_image(media_id):
        media = GraphApi.get_media_url(media_id)
        return GraphApi.download_media(media.get("url"))
