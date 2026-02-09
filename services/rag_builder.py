import hashlib
import json
import logging
import os
import re
import time
from typing import List, Optional, Union

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai
import redis
from redis.cluster import RedisCluster

from services.config import Config

_logger = logging.getLogger("rag_builder")

COLLECTION_NAME = Config.chroma_collection_name

_DEFAULT_TOP_K = 3
_DEFAULT_DISTANCE_THRESHOLD = 0.35
_VALID_CROP_CACHE_TTL_SECONDS = 300

# Embedding cache
_EMBED_CACHE_TTL_SECONDS = int(os.getenv("EMBED_CACHE_TTL_SECONDS", "21600"))  # 6 hours default
_EMBED_CACHE_PREFIX = "emb:v1"
_EMBED_NORM_MAXLEN = 512

# Logging controls
_EMBED_CACHE_LOG_EVERY_N_CALLS = int(os.getenv("EMBED_CACHE_LOG_EVERY_N_CALLS", "20"))  # info log throttle
_embed_call_counter = 0

# Very light normalization: keep meaning, improve cache hits
_FILLER_WORDS = {
    "how", "to", "please", "plz", "kindly", "tell", "me", "can", "you", "what", "is", "are",
    "the", "a", "an", "in", "on", "for", "of", "my", "i", "we"
}
_PUNCT_RE = re.compile(r"[^\w\s|]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

_gemini_client = None
_chroma_client = None
_collection = None
_valid_crop_cache = {"values": None, "fetched_at": 0.0}

_redis_client: Optional[Union[redis.Redis, RedisCluster]] = None
_redis_status_logged = False

def _get_redis_client() -> Optional[Union[redis.Redis, RedisCluster]]:
    """
    Returns a redis client for embedding cache:
      - USE_LOCAL_REDIS=false (default): RedisCluster (Azure Managed Redis cluster)
      - USE_LOCAL_REDIS=true:  redis.Redis (single node local)
    If Redis is unreachable, returns None and disables cache.
    """
    global _redis_client, _redis_status_logged

    if _redis_client is not None:
        return _redis_client

    use_local = os.getenv("USE_LOCAL_REDIS", "false").lower() == "true"

    host = Config.redis_host
    port = Config.redis_port
    password = Config.redis_password
    ssl_enabled = Config.redis_ssl

    try:
        if use_local:
            _logger.info("[REDIS] Using LOCAL single-node Redis for embedding cache")
            _redis_client = redis.Redis(
                host=host,
                port=port,
                password=password,
                ssl=ssl_enabled,
                decode_responses=False,
                socket_timeout=2,
            )
        else:
            _logger.info("[REDIS] Using CLUSTER Redis (Azure Managed) for embedding cache")
            _redis_client = RedisCluster(
                host=host,
                port=port,
                password=password,
                ssl=ssl_enabled,
                decode_responses=False,
                socket_timeout=2,
                skip_full_coverage_check=True,
            )

        _redis_client.ping()
        return _redis_client

    except Exception as exc:
        if not _redis_status_logged:
            _logger.warning(
                "Redis cache unavailable at %s:%s (ssl=%s, local=%s); embedding cache disabled: %s",
                host,
                port,
                ssl_enabled,
                use_local,
                exc,
            )
            _redis_status_logged = True

        _redis_client = None
        return None


def _normalize_for_embed_cache(text: str) -> str:
    """
    Light normalization to increase cache hits safely:
    - lowercase
    - strip punctuation (keep letters/numbers/_)
    - collapse whitespace
    - remove common filler words (keeps technical keywords like control/dosage/etc.)
    """
    if not text:
        return ""

    t = text.strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()

    if not t:
        return ""

    parts = [p for p in t.split(" ") if p and p not in _FILLER_WORDS]
    t = " ".join(parts).strip()

    if len(t) > _EMBED_NORM_MAXLEN:
        t = t[:_EMBED_NORM_MAXLEN]
    return t


def _embed_cache_key(model_name: str, normalized_text: str) -> str:
    digest = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
    return f"{_EMBED_CACHE_PREFIX}:{model_name}:{digest}"


def _serialize_embedding(vec: List[float]) -> bytes:
    return json.dumps(vec, separators=(",", ":")).encode("utf-8")


def _deserialize_embedding(raw: bytes) -> List[float]:
    return json.loads(raw.decode("utf-8"))


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    EmbeddingFunction used by Chroma for query_texts.
    Adds Redis caching with light normalization to reduce Gemini embed calls.
    """

    def __init__(self, client, model_name="text-embedding-004"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input_texts: Documents) -> Embeddings:
        global _embed_call_counter

        texts = list(input_texts or [])
        if not texts:
            return []

        _embed_call_counter += 1
        call_id = _embed_call_counter

        r = _get_redis_client()

        normalized = [_normalize_for_embed_cache(t) for t in texts]
        keys = [_embed_cache_key(self.model_name, n) if n else "" for n in normalized]

        cached: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        # --- Redis read ---
        t0 = time.perf_counter()
        redis_ms = None
        hit_count = 0

        if r is not None:
            try:
                # only non-empty keys
                non_empty_keys = [k for k in keys if k]
                raw_vals = r.mget(non_empty_keys) if non_empty_keys else []
                it = iter(raw_vals)

                for i, k in enumerate(keys):
                    if not k:
                        missing_indices.append(i)
                        continue

                    raw = next(it)
                    if raw:
                        try:
                            cached[i] = _deserialize_embedding(raw)
                            hit_count += 1
                        except Exception:
                            cached[i] = None
                            missing_indices.append(i)
                    else:
                        missing_indices.append(i)

                redis_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                _logger.warning("Redis mget failed; embedding cache bypassed for this call: %s", exc)
                missing_indices = list(range(len(texts)))
        else:
            missing_indices = list(range(len(texts)))

        miss_count = len(missing_indices)

        # Throttled INFO summary + always DEBUG details
        if (call_id % _EMBED_CACHE_LOG_EVERY_N_CALLS) == 0:
            _logger.info(
                "Embed cache summary (every %d calls): model=%s texts=%d hits=%d misses=%d ttl=%ss redis=%s",
                _EMBED_CACHE_LOG_EVERY_N_CALLS,
                self.model_name,
                len(texts),
                hit_count,
                miss_count,
                _EMBED_CACHE_TTL_SECONDS,
                "on" if r is not None else "off",
            )

        _logger.debug(
            "Embed cache call=%d model=%s texts=%d hits=%d misses=%d redis_read_ms=%.2f",
            call_id,
            self.model_name,
            len(texts),
            hit_count,
            miss_count,
            redis_ms if redis_ms is not None else -1.0,
        )

        # --- Gemini embed for misses only ---
        gemini_ms = 0.0
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            t1 = time.perf_counter()
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=missing_texts
                )
                new_embeds = [e.values for e in result.embeddings]
                gemini_ms = (time.perf_counter() - t1) * 1000.0
            except Exception as exc:
                _logger.exception("Gemini embedding failed: %s", exc)
                raise

            _logger.debug(
                "Gemini embed call=%d model=%s miss_count=%d gemini_ms=%.2f",
                call_id,
                self.model_name,
                len(missing_texts),
                gemini_ms,
            )

            # --- Redis write for new embeds ---
            if r is not None:
                t2 = time.perf_counter()
                try:
                    pipe = r.pipeline()
                    for idx, vec in zip(missing_indices, new_embeds):
                        cached[idx] = vec
                        k = keys[idx]
                        if k:
                            pipe.setex(k, _EMBED_CACHE_TTL_SECONDS, _serialize_embedding(vec))
                    pipe.execute()
                    redis_write_ms = (time.perf_counter() - t2) * 1000.0
                    _logger.debug(
                        "Redis embed cache write call=%d keys=%d write_ms=%.2f",
                        call_id,
                        len(missing_indices),
                        redis_write_ms,
                    )
                except Exception as exc:
                    _logger.warning("Redis cache write failed (continuing without): %s", exc)
                    for idx, vec in zip(missing_indices, new_embeds):
                        cached[idx] = vec
            else:
                for idx, vec in zip(missing_indices, new_embeds):
                    cached[idx] = vec

        # Output embeddings in order
        out: Embeddings = []
        for vec in cached:
            out.append(vec if vec is not None else [])
        return out

    def name(self) -> str:
        return "GeminiEmbeddingFunction"


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=Config.gemini_api_key)
    return _gemini_client


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    gemini_client = _get_gemini_client()
    embedding_fn = GeminiEmbeddingFunction(client=gemini_client)

    _chroma_client = chromadb.HttpClient(
        host=Config.chroma_host,
        port=Config.chroma_port,
        tenant=Config.chroma_tenant,
        database=Config.chroma_database,
        ssl=Config.chroma_ssl,
        headers=Config.chroma_headers,
    )

    try:
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except chromadb.errors.NotFoundError:
        _logger.error("Chroma collection missing: %s", COLLECTION_NAME)
        _collection = None
        return None

    _logger.info(
        "Chroma connected: %s:%s tenant=%s db=%s collection=%s",
        Config.chroma_host,
        Config.chroma_port,
        Config.chroma_tenant,
        Config.chroma_database,
        COLLECTION_NAME,
    )
    return _collection


def _get_valid_crops(collection):
    now = time.time()
    cached = _valid_crop_cache.get("values")
    if cached is not None and (now - _valid_crop_cache.get("fetched_at", 0)) < _VALID_CROP_CACHE_TTL_SECONDS:
        return cached

    try:
        metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    except Exception as exc:
        _logger.warning("Failed to load crop metadata census: %s", exc)
        return set()

    crops = set()
    for meta in metadatas:
        if not meta:
            continue
        crop = meta.get("crop")
        if crop:
            crops.add(str(crop).strip())

    _valid_crop_cache["values"] = crops
    _valid_crop_cache["fetched_at"] = now
    return crops


def _normalize_crop_tag(crop_name):
    return (crop_name or "").strip().lower().replace(" ", "_")


def _resolve_crop_tag(normalized_crop, valid_crops):
    if not normalized_crop:
        return ""

    if not valid_crops:
        return normalized_crop

    if normalized_crop in valid_crops:
        return normalized_crop

    for db_crop in valid_crops:
        if db_crop.startswith(f"{normalized_crop}_"):
            return db_crop

    return normalized_crop


def retrieve_rag_evidence(
    decomposed_queries,
    *,
    top_k=_DEFAULT_TOP_K,
    distance_threshold=_DEFAULT_DISTANCE_THRESHOLD
):
    if not decomposed_queries:
        return []

    collection = _get_collection()
    if collection is None:
        _logger.warning("RAG collection unavailable; returning empty results")
        results = []
        for line in decomposed_queries:
            if not isinstance(line, str) or "|" not in line:
                continue
            crop_raw, atomic_query = (part.strip() for part in line.split("|", 1))
            if not atomic_query:
                continue
            results.append({
                "query": atomic_query,
                "crop": crop_raw,
                "status": "ERROR",
                "evidence": [],
                "matched_crop": "",
                "score": 1.0,
            })
        return results

    valid_crops = _get_valid_crops(collection)

    parsed = []
    for line in decomposed_queries:
        if not isinstance(line, str) or "|" not in line:
            continue
        crop_raw, atomic_query = (part.strip() for part in line.split("|", 1))
        if not atomic_query:
            continue
        normalized_crop = _normalize_crop_tag(crop_raw)
        search_crop_tag = _resolve_crop_tag(normalized_crop, valid_crops)
        parsed.append({
            "crop": crop_raw,
            "query": atomic_query,
            "search_tag": search_crop_tag,
        })

    if not parsed:
        return []

    grouped = {}
    for entry in parsed:
        grouped.setdefault(entry["search_tag"], []).append(entry)

    results = []
    for tag, entries in grouped.items():
        if not tag:
            for entry in entries:
                results.append({
                    "query": entry["query"],
                    "crop": entry["crop"],
                    "status": "MISSING",
                    "evidence": [],
                    "matched_crop": "",
                    "score": 1.0,
                })
            continue

        try:
            response = collection.query(
                query_texts=[e["query"] for e in entries],
                n_results=top_k,
                where={"crop": tag},
            )
        except Exception as exc:
            _logger.exception("Chroma query failed for tag=%s: %s", tag, exc)
            for entry in entries:
                results.append({
                    "query": entry["query"],
                    "crop": entry["crop"],
                    "status": "ERROR",
                    "evidence": [],
                    "matched_crop": tag,
                    "score": 1.0,
                })
            continue

        documents = response.get("documents") or []
        distances = response.get("distances") or []

        for idx, entry in enumerate(entries):
            docs = documents[idx] if idx < len(documents) else []
            dists = distances[idx] if idx < len(distances) else []
            top_distance = dists[0] if dists else 1.0
            has_local_data = bool(docs)
            status = "FOUND" if has_local_data and top_distance < distance_threshold else "MISSING"

            raw_evidence = docs if status == "FOUND" else []
            clean_evidence = list(dict.fromkeys(raw_evidence))

            results.append({
                "query": entry["query"],
                "crop": entry["crop"],
                "status": status,
                "evidence": clean_evidence,
                "matched_crop": tag,
                "score": float(top_distance),
            })

    return results


def warm_rag_cache():
    try:
        collection = _get_collection()
        _get_valid_crops(collection)
    except Exception as exc:
        _logger.warning("RAG warmup failed: %s", exc)
        return False
    return True

def list_chroma_collections():
    """Diagnostic helper to list all collections and their item counts."""
    # Ensure the client is initialized
    _get_collection() 
    
    global _chroma_client
    if _chroma_client is None:
        return {"error": "Chroma client not initialized"}
    
    try:
        collections = _chroma_client.list_collections()
        result = []
        for col in collections:
            # We fetch the count for each to be sure data exists
            count = col.count()
            result.append({
                "name": col.name,
                "count": count,
                "metadata": col.metadata
            })
        return {"collections": result, "total_count": len(result)}
    except Exception as e:
        _logger.error(f"Failed to list collections: {e}")
        return {"error": str(e)}
