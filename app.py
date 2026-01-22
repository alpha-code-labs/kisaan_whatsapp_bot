import hashlib
import hmac
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from services.config import Config
from services.conversation import Conversation
from services.rag_builder import warm_rag_cache

app = FastAPI()
logger = logging.getLogger("app")


@app.on_event("startup")
def on_startup():
    Config.check_env_variables()
    Config.print_config()
    if not warm_rag_cache():
        logger.warning("RAG warmup did not complete; first request may be slower")


@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode != "subscribe" or token != Config.verify_token:
        raise HTTPException(status_code=403, detail="Forbidden")

    return PlainTextResponse(challenge or "")


@app.post("/webhook")
async def handle_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")

    try:
        verify_request_signature(raw_body, signature)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if payload.get("object") == "whatsapp_business_account":
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value") or {}
                sender_phone_number_id = value.get("metadata", {}).get("phone_number_id")

                for status in value.get("statuses", []) or []:
                    Conversation.handle_status(sender_phone_number_id, status)

                for raw_message in value.get("messages", []) or []:
                    Conversation.handle_message(sender_phone_number_id, raw_message)

    return PlainTextResponse("EVENT_RECEIVED")


@app.get("/")
async def health_check():
    return JSONResponse({
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
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=Config.port, reload=False)
