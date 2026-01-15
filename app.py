import hashlib
import hmac
from flask import Flask, request, jsonify

from services.config import Config
from services.conversation import Conversation

app = Flask(__name__)
Config.check_env_variables()
Config.print_config()


@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode != "subscribe" or token != Config.verify_token:
        return ("Forbidden", 403)

    return challenge or ""


@app.post("/webhook")
def handle_webhook():
    raw_body = request.get_data()
    signature = request.headers.get("x-hub-signature-256")

    try:
        verify_request_signature(raw_body, signature)
    except ValueError:
        return ("Invalid signature", 403)

    payload = request.get_json(silent=True) or {}

    if payload.get("object") == "whatsapp_business_account":
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value") or {}
                sender_phone_number_id = value.get("metadata", {}).get("phone_number_id")

                for status in value.get("statuses", []) or []:
                    Conversation.handle_status(sender_phone_number_id, status)

                for raw_message in value.get("messages", []) or []:
                    Conversation.handle_message(sender_phone_number_id, raw_message)

    return ("EVENT_RECEIVED", 200)


@app.get("/")
def health_check():
    return jsonify({
        "message": "Kisaan bot server is running",
        "endpoints": ["POST /webhook - WhatsApp webhook endpoint"]
    })


def verify_request_signature(raw_body, signature_header):
    if not signature_header:
        print("WARNING: Missing x-hub-signature-256 header")
        return

    try:
        _, signature_hash = signature_header.split("=", 1)
    except ValueError:
        raise ValueError("Invalid signature header")

    expected = hmac.new(
        Config.app_secret.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256
    ).hexdigest()

    if signature_hash != expected:
        raise ValueError("Signature mismatch")


if __name__ == "__main__":
    Config.check_env_variables()
    app.run(host="0.0.0.0", port=Config.port)
