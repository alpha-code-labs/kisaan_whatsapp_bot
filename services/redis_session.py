import json
import redis
from services.config import Config

SESSION_TTL = 30

SessionState = {
    "GREETING": "GREETING",
    "AWAITING_MENU_WEATHER_CHOICE": "AWAITING_MENU_WEATHER_CHOICE",
    "AWAITING_MENU_CROP_ADVICE_CHOICE": "AWAITING_MENU_CROP_ADVICE_CHOICE",
    "AWAITING_WEATHER_LOCATION": "AWAITING_WEATHER_LOCATION",
    "AWAITING_DISTRICT_NAME": "AWAITING_DISTRICT_NAME",
    "AWAITING_CROP_NAME": "AWAITING_CROP_NAME",
    "WEATHER": "WEATHER",
    "CROP_ADVICE_CATEGORY_MENU": "CROP_ADVICE_CATEGORY_MENU",
    "CROP_ADVICE_QUERY": "CROP_ADVICE_QUERY",
    "CROP_ADVICE_QUERY_COLLECTING": "CROP_ADVICE_QUERY_COLLECTING",
    "CROP_ADVICE_QUERY_CONFIRM": "CROP_ADVICE_QUERY_CONFIRM",
    "PROCESSING_CROP_QUERY": "PROCESSING_CROP_QUERY"
}

_client = redis.Redis(
    host=Config.redis_host,
    port=Config.redis_port,
    decode_responses=True
)


def get_session(user_id):
    data = _client.get(f"session:{user_id}")
    return json.loads(data) if data else None


def create_session(user_id):
    session = {
        "userId": user_id,
        "state": SessionState["GREETING"],
        "queryType": None,
        "location": None,
        "district": None,
        "isExistingCrop": False,
        "cropAdviceCategory": None,
        "crop": None,
        "query": {"texts": [], "audios": [], "images": []},
        "createdAt": _now_ms(),
        "updatedAt": _now_ms()
    }

    _client.setex(f"session:{user_id}", SESSION_TTL, json.dumps(session))
    return session


def update_session(user_id, updates):
    session = get_session(user_id)
    if not session:
        return create_session(user_id)

    updated = {**session, **updates, "updatedAt": _now_ms()}
    _client.setex(f"session:{user_id}", SESSION_TTL, json.dumps(updated))
    return updated


def update_session_state(user_id, new_state):
    if new_state not in SessionState.values():
        raise ValueError("Invalid session state")

    update = {"state": new_state}
    if new_state == SessionState["WEATHER"]:
        update["queryType"] = "WEATHER"
    if new_state == SessionState["CROP_ADVICE_CATEGORY_MENU"]:
        update["queryType"] = "CROP_ADVICE"
    return update_session(user_id, update)


def update_crop_advice_category(user_id, category):
    return update_session(user_id, {"cropAdviceCategory": category})


def update_crop_info(user_id, crop):
    return update_session(user_id, {"crop": crop})

def update_is_Existing_Crop(user_id, is_existing_crop):
    return update_session(user_id, {"is_existing_crop": is_existing_crop})

def update_district_info(user_id, district):
    return update_session(user_id, {"district": district})

def update_user_query(user_id, query):
    return update_session(user_id, {"query": query})


def set_user_location(user_id, location):
    return update_session(user_id, {"location": location})


def append_user_query(user_id, payload):
    session = get_session(user_id) or create_session(user_id)
    query = session.get("query") or {"texts": [], "audios": [], "images": []}

    if payload.get("text"):
        query["texts"].append(payload["text"])
    if payload.get("audioId"):
        query["audios"].append(payload["audioId"])
    if payload.get("imageId"):
        query["images"].append(payload["imageId"])

    return update_session(user_id, {"query": query})


def delete_session(user_id):
    _client.delete(f"session:{user_id}")


def _now_ms():
    import time
    return int(time.time() * 1000)
