import hashlib
import hmac
import logging
import inspect
from functools import partial
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from services.config import Config
from services.conversation import Conversation
from services.rag_builder import warm_rag_cache, list_chroma_collections

logger = logging.getLogger("app")


async def _call_maybe_async(fn, *args, **kwargs):
    """
    Call fn in the safest non-blocking way:
    - If fn is async (or returns an awaitable), await it.
    - Otherwise, run it in a worker thread to avoid blocking the event loop.

    This keeps behavior identical today and becomes naturally async once you
    convert underlying functions to async later.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)

    try:
        result = fn(*args, **kwargs)
    except TypeError:
        # Some callables may not accept kwargs in certain wrappers; fall back.
        result = await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))
        return result

    if inspect.isawaitable(result):
        return await result

    # If it's a normal sync result, but the function already executed on the loop,
    # we still want to avoid blocking for heavier calls. So run sync functions in
    # a thread by default.
    return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting initialization...")
        Config.check_env_variables()
        # If Config uses os.environ['KEY'] and it's missing on Azure,
        # the app crashes here.
        yield
    except Exception as e:
        logger.error(f"CRITICAL CRASH DURING LIFESPAN: {str(e)}", exc_info=True)
        raise e


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/chroma")
async def debug_chroma():
    """
    Diagnostic endpoint to verify ChromaDB connectivity
    and see if your persistent data is actually mounted.
    """
    data = await _call_maybe_async(list_chroma_collections)
    return JSONResponse(content=data)


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
                    await _call_maybe_async(Conversation.handle_status, sender_phone_number_id, status)

                for raw_message in value.get("messages", []) or []:
                    await _call_maybe_async(Conversation.handle_message, sender_phone_number_id, raw_message)

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
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)