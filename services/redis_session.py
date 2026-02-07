import json
import os
import redis
import uuid
from services.config import Config
from redis.cluster import RedisCluster

SESSION_TTL = 300

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
    "PROCESSING_CROP_QUERY": "PROCESSING_CROP_QUERY",
    "AWAITING_AMBIGUOUS_CROP_CHOICE": "AWAITING_AMBIGUOUS_CROP_CHOICE",
    "AWAITING_CROP_CONFIRMATION": "AWAITING_CROP_CONFIRMATION",
    "QUERY_NOT_ABOUT_CROP": "QUERY_NOT_ABOUT_CROP",
}

# Azure Managed Redis in Cluster Mode
_client = RedisCluster(
    host=Config.redis_host,
    port=Config.redis_port,
    password=Config.redis_password,
    ssl=Config.redis_ssl,
    decode_responses=True,
    skip_full_coverage_check=True  # Recommended for Azure Managed Redis
)


def get_session(user_id):
    data = _client.get(f"session:{user_id}")
    session = json.loads(data) if data else None
    return session


def create_session(user_id):
    session = {
        "userId": user_id,
        "sessionId": uuid.uuid4().hex[:8],
        "state": SessionState["GREETING"],
        "queryType": None,
        "location": None,
        "district": None,
        "isExistingCrop": False,
        "cropAdviceCategory": None,
        "crop": None,
        "query": {"texts": [], "audios": [], "images": []},
        "uploadCount": 0,
        "geminiAggregatedQuery": None,
        "geminiAggregatedQueryDecomposed": [],
        "geniminiSplittedQueries" : [],
        "adviceResponses": [],
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

    session = update_session(user_id, update)

    # 🔍 DEBUG LOG
    print(
        f"[SESSION] STATE_UPDATE | user={user_id} | "
        f"new_state={new_state} | "
        f"crop={session.get('crop')} | "
        f"isExistingCrop={session.get('isExistingCrop')} | "
        f"category={session.get('cropAdviceCategory')} | "
    )

    return session



def update_crop_advice_category(user_id, category):
    return update_session(user_id, {"cropAdviceCategory": category})


def update_crop_info(user_id, crop):
    return update_session(user_id, {"crop": crop})

def update_is_existing_crop(user_id, is_existing_crop: bool):
    return update_session(user_id, {"isExistingCrop": bool(is_existing_crop)})

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
    if payload.get("audioUrl"):
        query["audios"].append(payload["audioUrl"])
    if payload.get("imageUrl"):
        query["images"].append(payload["imageUrl"])

    return update_session(user_id, {"query": query})


def delete_session(user_id):
    _client.delete(f"session:{user_id}")


def dump_session(user_id, failed = False):
    session = get_session(user_id)
    if not session:
        return

    session_id = session.get("sessionId") or "unknown"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sessions_dir = os.path.join(base_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    if not failed:
        path = os.path.join(sessions_dir, f"{user_id}_{session_id}.json")
    else:
        path = os.path.join(sessions_dir, f"{user_id}_{session_id}_failed.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=True, indent=2)


def append_advice_response(user_id, response_text):
    session = get_session(user_id) or create_session(user_id)
    responses = session.get("adviceResponses") or []
    responses.append(response_text)
    return update_session(user_id, {"adviceResponses": responses})

def append_aggregated_query_response(user_id, response_text):
    session = get_session(user_id) or create_session(user_id)
    return update_session(user_id, {"geminiAggregatedQuery": response_text})

def append_aggregated_query_decomposed_response(user_id, response_text):
    session = get_session(user_id) or create_session(user_id)
    return update_session(user_id, {"geminiAggregatedQueryDecomposed": response_text})


def reset_query_arrays(user_id):
    session = get_session(user_id) or create_session(user_id)
    query = {"texts": [], "audios": [], "images": []}
    return update_session(user_id, {"query": query, "uploadCount": 0})


def next_upload_count(user_id):
    session = get_session(user_id) or create_session(user_id)
    count = int(session.get("uploadCount", 0)) + 1
    update_session(user_id, {"uploadCount": count})
    return count


def _now_ms():
    import time
    return int(time.time() * 1000)
