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
                    "text": "Please! Choose a category to get started."
                },
                "action": {
                    "button": "Choose Category",
                    "sections": [
                        {
                            "title": "Categories",
                            "rows": [
                                {"id": "weather_info", "title": "Weather"},
                                {"id": "disease_management", "title": "Disease management"},
                                {"id": "insect_management", "title": "Insect management"},
                                {"id": "fertilizer_use", "title": "Fertilizer use"},
                                {"id": "weed_management", "title": "Weed management"},
                                {"id": "variety_sowing_time", "title": "Varieties & Sowing Time"},
                                {"id": "others", "title": "Others"}
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
                    "text": 'क्या आप और जानकारी जोड़ना चाहते हैं या आपकी जानकारी पूरी हो गई है?'
                },
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {"id": "query_continue", "title": "➕ और जानकारी जोड़ें"}
                        },
                        {
                            "type": "reply",
                            "reply": {"id": "query_done", "title": "✅ जानकारी पूरी हो गई है"}
                        }
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
                "body": {
                    "text": "कृपया विशेषज्ञ मार्गदर्शन के लिए एक श्रेणी चुनें:"
                },
                "footer": {"text": "नीचे दी गई किसी श्रेणी पर टैप करें"},
                "action": {
                    "button": "श्रेणियाँ देखें",
                    "sections": [
                        {
                            "title": "उपलब्ध श्रेणियाँ",
                            "rows": [
                                {"id": "disease_management", "title": "रोग प्रबंधन"},
                                {"id": "insect_management", "title": "कीट प्रबंधन"},
                                {"id": "fertilizer_use", "title": "उर्वरक का उपयोग"},
                                {"id": "weed_management", "title": "बाली प्रबंधन"},
                                {"id": "variety", "title": "किस्में"},
                                {"id": "sowing_time", "title": "बोने का समय"},
                                {"id": "others", "title": "अन्य"}
                            ]
                        }
                    ]
                }
            }
        }

        return GraphApi._make_api_call(message_id, sender_phone_number_id, request_body)

    
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
                "body": {
                    "text": f"क्या आप {crop_name_hi} के बारे में जानना चाहते हैं?"
                },
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
