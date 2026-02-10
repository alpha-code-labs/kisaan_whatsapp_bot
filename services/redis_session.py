import json
import os
import uuid
from services.config import Config

from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster

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

USE_LOCAL = os.getenv("USE_LOCAL_REDIS", "false").lower() == "true"

_client = None

if USE_LOCAL:
    print("[REDIS] Using LOCAL single-node Redis")

    _client = Redis(
        host=Config.redis_host,
        port=Config.redis_port,
        password=Config.redis_password,
        ssl=Config.redis_ssl,
        decode_responses=True
    )

else:
    print("[REDIS] Using CLUSTER Redis (Azure Managed)")

    _client = RedisCluster(
        host=Config.redis_host,
        port=Config.redis_port,
        password=Config.redis_password,
        ssl=Config.redis_ssl,
        decode_responses=True
    )


async def get_session(user_id):
    data = await _client.get(f"session:{user_id}")
    session = json.loads(data) if data else None
    return session


async def create_session(user_id):
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
        "geniminiSplittedQueries": [],
        "adviceResponses": [],
        "createdAt": _now_ms(),
        "updatedAt": _now_ms()
    }

    await _client.setex(f"session:{user_id}", SESSION_TTL, json.dumps(session))
    return session


async def update_session(user_id, updates):
    session = await get_session(user_id)
    if not session:
        return await create_session(user_id)

    updated = {**session, **updates, "updatedAt": _now_ms()}
    await _client.setex(f"session:{user_id}", SESSION_TTL, json.dumps(updated))
    return updated


async def update_session_state(user_id, new_state):
    if new_state not in SessionState.values():
        raise ValueError("Invalid session state")

    update = {"state": new_state}
    if new_state == SessionState["WEATHER"]:
        update["queryType"] = "WEATHER"
    if new_state == SessionState["CROP_ADVICE_CATEGORY_MENU"]:
        update["queryType"] = "CROP_ADVICE"

    session = await update_session(user_id, update)

    # 🔍 DEBUG LOG (unchanged)
    print(
        f"[SESSION] STATE_UPDATE | user={user_id} | "
        f"new_state={new_state} | "
        f"crop={session.get('crop')} | "
        f"isExistingCrop={session.get('isExistingCrop')} | "
        f"category={session.get('cropAdviceCategory')} | "
    )

    return session


async def update_crop_advice_category(user_id, category):
    return await update_session(user_id, {"cropAdviceCategory": category})


async def update_crop_info(user_id, crop):
    return await update_session(user_id, {"crop": crop})


async def update_is_existing_crop(user_id, is_existing_crop: bool):
    return await update_session(user_id, {"isExistingCrop": bool(is_existing_crop)})


async def update_district_info(user_id, district):
    return await update_session(user_id, {"district": district})


async def update_user_query(user_id, query):
    return await update_session(user_id, {"query": query})


async def set_user_location(user_id, location):
    return await update_session(user_id, {"location": location})


async def append_user_query(user_id, payload):
    session = await get_session(user_id) or await create_session(user_id)
    query = session.get("query") or {"texts": [], "audios": [], "images": []}

    if payload.get("text"):
        query["texts"].append(payload["text"])
    if payload.get("audioUrl"):
        query["audios"].append(payload["audioUrl"])
    if payload.get("imageUrl"):
        query["images"].append(payload["imageUrl"])

    return await update_session(user_id, {"query": query})


async def delete_session(user_id):
    await _client.delete(f"session:{user_id}")


async def dump_session(user_id, failed=False):
    """
    Keeps same behavior: writes JSON to Config.sessions_dir.
    File I/O is still blocking; we keep it identical here.
    We'll convert this to aiofiles later if you want, but it's not required for correctness.
    """
    session = await get_session(user_id)
    if not session:
        return

    session_id = session.get("sessionId") or "unknown"

    sessions_dir = Config.sessions_dir
    os.makedirs(sessions_dir, exist_ok=True)

    suffix = "_failed" if failed else ""
    path = os.path.join(sessions_dir, f"{user_id}_{session_id}{suffix}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=True, indent=2)


async def append_advice_response(user_id, response_text):
    session = await get_session(user_id) or await create_session(user_id)
    responses = session.get("adviceResponses") or []
    responses.append(response_text)
    return await update_session(user_id, {"adviceResponses": responses})


async def append_aggregated_query_response(user_id, response_text):
    _ = await get_session(user_id) or await create_session(user_id)
    return await update_session(user_id, {"geminiAggregatedQuery": response_text})


async def append_aggregated_query_decomposed_response(user_id, response_text):
    _ = await get_session(user_id) or await create_session(user_id)
    return await update_session(user_id, {"geminiAggregatedQueryDecomposed": response_text})


async def reset_query_arrays(user_id):
    _ = await get_session(user_id) or await create_session(user_id)
    query = {"texts": [], "audios": [], "images": []}
    return await update_session(user_id, {"query": query, "uploadCount": 0})


async def next_upload_count(user_id):
    session = await get_session(user_id) or await create_session(user_id)
    count = int(session.get("uploadCount", 0)) + 1
    await update_session(user_id, {"uploadCount": count})
    return count


def _now_ms():
    import time
    return int(time.time() * 1000)


async def mark_incoming_message_seen(message_id: str, ttl_s: int = 3600) -> bool:
    """
    Returns True if this message_id is seen for the first time.
    Returns False if we've already processed it recently.
    """
    if not message_id:
        return True  # can't dedupe

    key = f"seen:wa:msg:{message_id}"
    try:
        # SET NX EX = atomic idempotency lock
        ok = await _client.set(key, "1", nx=True, ex=int(ttl_s))
        return bool(ok)
    except Exception:
        # If Redis is down, don't break the bot; process normally
        return True