import logging
import time
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai

from services.config import Config

_logger = logging.getLogger("rag_builder")

DB_DIR = Path(Config.chroma_db_dir)
COLLECTION_NAME = Config.chroma_collection_name

_DEFAULT_TOP_K = 3
_DEFAULT_DISTANCE_THRESHOLD = 0.35
_VALID_CROP_CACHE_TTL_SECONDS = 300

_gemini_client = None
_chroma_client = None
_collection = None
_valid_crop_cache = {"values": None, "fetched_at": 0.0}


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client, model_name="text-embedding-004"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input_texts: Documents) -> Embeddings:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=input_texts
        )
        return [e.values for e in result.embeddings]

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
    _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except chromadb.errors.NotFoundError as exc:
        _logger.error("Chroma collection missing: %s", COLLECTION_NAME)
        _collection = None
        return None

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
    """
    Retrieve evidence for decomposed queries.
    Each entry in decomposed_queries is expected to be "Crop | question".
    Returns a list of dicts with query, crop, status, evidence, matched_crop, score.
    """
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
    """
    Initialize Chroma collection and cached crop metadata.
    Returns True on success, False otherwise.
    """
    try:
        collection = _get_collection()
        _get_valid_crops(collection)
    except Exception as exc:
        _logger.warning("RAG warmup failed: %s", exc)
        return False
    return True
