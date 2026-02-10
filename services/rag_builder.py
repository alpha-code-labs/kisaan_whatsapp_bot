import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import List, Optional, Union, Any, Dict

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai

from services.config import Config

_logger = logging.getLogger("rag_builder")

COLLECTION_NAME = Config.chroma_collection_name

_DEFAULT_TOP_K = 3
_DEFAULT_DISTANCE_THRESHOLD = 0.35
_VALID_CROP_CACHE_TTL_SECONDS = 300

_PUNCT_RE = re.compile(r"[^\w\s|]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

_gemini_client = None
_chroma_client = None
_collection = None
_valid_crop_cache = {"values": None, "fetched_at": 0.0}


def _normalize_for_embed(text: str) -> str:
    """
    Lightweight normalization to reduce trivial embedding differences.
    (No Redis cache; this is only for stable hashing/logging if needed.)
    """
    if not text:
        return ""
    t = text.strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Chroma EmbeddingFunction interface is SYNC.

    ✅ Redis caching REMOVED completely.
    We embed all input texts directly via Gemini embed_content (sync SDK),
    and we run the overall Chroma pipeline in a worker thread (see retrieve_rag_evidence).
    """

    def __init__(self, client, model_name: str = "gemini-embedding-001"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input_texts: Documents) -> Embeddings:
        texts = list(input_texts or [])
        if not texts:
            return []

        # Normalize only to avoid accidental None/whitespace weirdness
        cleaned = [_normalize_for_embed(t) for t in texts]

        t0 = time.perf_counter()
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=cleaned,
            )
            out = [e.values for e in result.embeddings]
        except Exception as exc:
            _logger.exception("Gemini embedding failed: %s", exc)
            raise

        gemini_ms = (time.perf_counter() - t0) * 1000.0
        _logger.debug(
            "Gemini embed model=%s texts=%d gemini_ms=%.2f",
            self.model_name,
            len(cleaned),
            gemini_ms,
        )

        # Chroma expects a list-of-vectors aligned with input_texts length.
        return out

    def name(self) -> str:
        return "GeminiEmbeddingFunction"


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=Config.gemini_api_key)
    return _gemini_client


def _get_collection_sync():
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


def _get_valid_crops_sync(collection):
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


def _retrieve_rag_evidence_sync(
    decomposed_queries,
    *,
    top_k=_DEFAULT_TOP_K,
    distance_threshold=_DEFAULT_DISTANCE_THRESHOLD,
):
    if not decomposed_queries:
        return []

    collection = _get_collection_sync()
    if collection is None:
        _logger.warning("RAG collection unavailable; returning empty results")
        results = []
        for line in decomposed_queries:
            if not isinstance(line, str) or "|" not in line:
                continue
            crop_raw, atomic_query = (part.strip() for part in line.split("|", 1))
            if not atomic_query:
                continue
            results.append(
                {
                    "query": atomic_query,
                    "crop": crop_raw,
                    "status": "ERROR",
                    "evidence": [],
                    "matched_crop": "",
                    "score": 1.0,
                }
            )
        return results

    valid_crops = _get_valid_crops_sync(collection)

    parsed = []
    for line in decomposed_queries:
        if not isinstance(line, str) or "|" not in line:
            continue
        crop_raw, atomic_query = (part.strip() for part in line.split("|", 1))
        if not atomic_query:
            continue
        normalized_crop = _normalize_crop_tag(crop_raw)
        search_crop_tag = _resolve_crop_tag(normalized_crop, valid_crops)
        parsed.append(
            {
                "crop": crop_raw,
                "query": atomic_query,
                "search_tag": search_crop_tag,
            }
        )

    if not parsed:
        return []

    grouped = {}
    for entry in parsed:
        grouped.setdefault(entry["search_tag"], []).append(entry)

    results = []
    for tag, entries in grouped.items():
        if not tag:
            for entry in entries:
                results.append(
                    {
                        "query": entry["query"],
                        "crop": entry["crop"],
                        "status": "MISSING",
                        "evidence": [],
                        "matched_crop": "",
                        "score": 1.0,
                    }
                )
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
                results.append(
                    {
                        "query": entry["query"],
                        "crop": entry["crop"],
                        "status": "ERROR",
                        "evidence": [],
                        "matched_crop": tag,
                        "score": 1.0,
                    }
                )
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

            results.append(
                {
                    "query": entry["query"],
                    "crop": entry["crop"],
                    "status": status,
                    "evidence": clean_evidence,
                    "matched_crop": tag,
                    "score": float(top_distance),
                }
            )

    return results


async def retrieve_rag_evidence(
    decomposed_queries,
    *,
    top_k=_DEFAULT_TOP_K,
    distance_threshold=_DEFAULT_DISTANCE_THRESHOLD,
):
    # Run the whole pipeline in a worker thread (Chroma + Gemini embed are sync).
    return await asyncio.to_thread(
        _retrieve_rag_evidence_sync,
        decomposed_queries,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )


async def warm_rag_cache():
    try:
        def _warm():
            collection = _get_collection_sync()
            if collection is None:
                return False
            _get_valid_crops_sync(collection)
            return True

        return await asyncio.to_thread(_warm)
    except Exception as exc:
        _logger.warning("RAG warmup failed: %s", exc)
        return False


async def list_chroma_collections():
    """Diagnostic helper to list all collections and their item counts."""
    def _list() -> Dict[str, Any]:
        _get_collection_sync()

        global _chroma_client
        if _chroma_client is None:
            return {"error": "Chroma client not initialized"}

        try:
            collections = _chroma_client.list_collections()
            result = []
            for col in collections:
                count = col.count()
                result.append(
                    {
                        "name": col.name,
                        "count": count,
                        "metadata": col.metadata,
                    }
                )
            return {"collections": result, "total_count": len(result)}
        except Exception as e:
            _logger.error(f"Failed to list collections: {e}")
            return {"error": str(e)}

    return await asyncio.to_thread(_list)