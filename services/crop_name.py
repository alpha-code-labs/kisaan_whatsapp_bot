import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from services.config import Config
from services.crop_detector import CropDetector, normalize_text
import logging
import uuid
from datetime import datetime

logger = logging.getLogger("crop_detect")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global lock to prevent concurrent file writes corruption
FILE_LOCK = threading.Lock()

client = genai.Client(api_key=Config.gemini_api_key)

_DETECTOR_CACHE: Dict[str, Any] = {
    "path": None,
    "mtime": None,
    "detector": None,
    "data": None,
}

def _load_crops_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"crops": [], "ambiguous_names": []}

def _get_detector_and_data() -> Tuple[CropDetector, Dict[str, Any], str]:
    crops_path = os.path.join(Config.data_dir, "crops.json")
    
    # Ensure directory exists
    os.makedirs(Config.data_dir, exist_ok=True)
    if not os.path.exists(crops_path):
        _atomic_write_json(crops_path, {"crops": [], "ambiguous_names": []})

    mtime = os.path.getmtime(crops_path)

    if (
        _DETECTOR_CACHE["detector"] is None
        or _DETECTOR_CACHE["path"] != crops_path
        or _DETECTOR_CACHE["mtime"] != mtime
    ):
        data = _load_crops_json(crops_path)
        detector = CropDetector(crops=data.get("crops", []), enable_transliteration=True)

        _DETECTOR_CACHE.update({
            "path": crops_path,
            "mtime": mtime,
            "detector": detector,
            "data": data
        })

    return _DETECTOR_CACHE["detector"], _DETECTOR_CACHE["data"], crops_path

def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    dir_name = os.path.dirname(path)
    # Use delete=False for os.replace compatibility on all systems
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def _clean_synonyms(master_name: str, synonyms: Optional[Any]) -> List[Dict[str, str]]:
    """
    Returns a safe synonyms list in the required format:
      [{"en": "...", "hi": "..."}, ...]
    Accepts messy inputs and converts when possible.
    """
    out: List[Dict[str, str]] = []

    def _add(en: str = "", hi: str = "") -> None:
        en = (en or "").strip()
        hi = (hi or "").strip()
        if not en and not hi:
            return
        out.append({"en": en, "hi": hi})

    # If None/empty -> fallback later
    if not synonyms:
        return []

    # Case 1: synonyms already a dict {"en": "...", "hi": "..."}
    if isinstance(synonyms, dict):
        _add(synonyms.get("en", ""), synonyms.get("hi", ""))
        return out

    # Case 2: synonyms is a single string
    if isinstance(synonyms, str):
        # treat as english/translit by default
        _add(synonyms, "")
        return out

    # Case 3: synonyms is a list (mixed types)
    if isinstance(synonyms, list):
        for s in synonyms:
            if isinstance(s, dict):
                _add(s.get("en", ""), s.get("hi", ""))
            elif isinstance(s, str):
                # convert string -> {"en": s, "hi": ""}
                _add(s, "")
            elif isinstance(s, (tuple, list)) and len(s) == 2:
                # convert ["wheat", "गेहूं"] -> {"en": "wheat", "hi": "गेहूं"}
                en, hi = s[0], s[1]
                _add(str(en), str(hi))
            else:
                # skip unknown types
                continue
        return out

    # Unknown type -> return empty
    return []


def _dedupe_synonyms(synonyms: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for s in synonyms:
        en = (s.get("en") or "").strip()
        hi = (s.get("hi") or "").strip()
        key = (normalize_text(en), normalize_text(hi))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"en": en, "hi": hi})
    return unique


def _find_ambiguous_match(query: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    entries = data.get("ambiguous_names") or []
    if not entries:
        return None

    q_norm = normalize_text(query)
    if not q_norm:
        return None

    q_tokens = set(q_norm.split())

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        variants: List[str] = []
        input_word = entry.get("input_word")
        if isinstance(input_word, dict):
            variants.extend([input_word.get("en", ""), input_word.get("hi", "")])
        elif isinstance(input_word, str):
            variants.append(input_word)

        variations = entry.get("variations") or []
        if isinstance(variations, list):
            variants.extend(variations)
        elif isinstance(variations, str):
            variants.append(variations)

        for v in variants:
            v_norm = normalize_text(str(v))
            if not v_norm:
                continue
            if " " in v_norm:
                if v_norm in q_norm:
                    return entry
            else:
                if v_norm in q_tokens:
                    return entry

    return None


def _pick_hindi_from_synonyms(synonyms: Optional[Any]) -> str:
    if isinstance(synonyms, list):
        for s in synonyms:
            if isinstance(s, dict):
                hi = (s.get("hi") or "").strip()
                if hi:
                    return hi
    elif isinstance(synonyms, dict):
        hi = (synonyms.get("hi") or "").strip()
        if hi:
            return hi
    return ""


def _get_hindi_name_for_master(master_name: str, data: Dict[str, Any]) -> str:
    master_norm = normalize_text(master_name)
    for item in data.get("crops", []):
        if not isinstance(item, dict):
            continue
        if normalize_text(item.get("master_name", "")) == master_norm:
            hi = _pick_hindi_from_synonyms(item.get("synonyms"))
            return hi or master_name
    return master_name


def _add_new_crop_to_file(
    crops_path: str,
    master_name: str,
    synonyms: Optional[Any] = None
) -> bool:
    master_name = (master_name or "").strip()
    if not master_name:
        return False

    with FILE_LOCK:
        data = _load_crops_json(crops_path)
        crops = data.get("crops", [])
        if not isinstance(crops, list):
            crops = []
            data["crops"] = crops

        # prevent duplicates
        if any(normalize_text((c or {}).get("master_name", "")) == normalize_text(master_name)
               for c in crops if isinstance(c, dict)):
            return False

        # clean + dedupe synonyms
        clean = _clean_synonyms(master_name, synonyms)
        clean = _dedupe_synonyms(clean)

        # guaranteed fallback (never write empty/bad synonyms)
        if not clean:
            clean = [{"en": master_name, "hi": ""}]

        crops.append({
            "master_name": master_name,
            "synonyms": clean,
        })

        _atomic_write_json(crops_path, data)

        # Clear cache to force reload on next call
        _DETECTOR_CACHE["detector"] = None
        return True

def _ai_detect_crop(query: str, master_names: List[str], trace_id: str) -> Dict[str, Any]:
    crop_list_str = ", ".join(master_names)

    system_instruction = f"""You are a Senior Agronomist in Haryana, India.
Identify the crop from user input.

MASTER LIST: {crop_list_str}

OUTPUT RULES (return ONLY one of these, no extra text):
A) If crop is found in MASTER LIST, return exactly:
<Exact Master Name>|found

B) If crop is a real crop but NOT in MASTER LIST, return VALID JSON only with this exact schema:
{{
  "master_name": "<Title Case Name>",
  "synonyms": [
    {{"en": "<English name or transliteration>", "hi": "<Hindi name>"}}
  ]
}}

Schema rules:
- "synonyms" MUST be a JSON array.
- Each item in "synonyms" MUST be an object with ONLY keys: "en" and "hi".
- Do NOT put strings in the synonyms array.
- If you know only one language, set the other to "" (empty string).
- Return at least 1 synonym object.

C) If input is not a crop, return exactly:
no crop found"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"{system_instruction}\n\nUser input: {query}",
            config={"temperature": 0}
        )
        raw = (response.text or "").strip()
        logger.info(f"[{trace_id}] LLM raw output: {raw!r}")
    except Exception as e:
        logger.exception(f"[{trace_id}] AI Detection Error")
        return {"status": "none", "raw_text": "", "error": str(e)}

    lower = raw.lower()

    # C) not a crop
    if lower.strip() == "no crop found" or "no crop found" in lower:
        return {"status": "none", "raw_text": raw}

    # A) found in master list: "<Exact Master Name>|found"
    if raw.endswith("|found"):
        crop_name = raw.rsplit("|found", 1)[0].strip()
        return {"status": "master", "crop_name": crop_name, "raw_text": raw}

    # B) JSON new crop
    json_part = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(json_part)
        return {
            "status": "new",
            "crop_name": parsed.get("master_name"),
            "synonyms": parsed.get("synonyms"),
            "raw_text": raw
        }
    except json.JSONDecodeError:
        logger.warning(f"[{trace_id}] LLM output not parseable as JSON and not |found.")
        return {"status": "none", "raw_text": raw}

def detect_crop(query: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
    trace_id = trace_id or uuid.uuid4().hex[:8]
    logger.info(f"[{trace_id}] detect_crop() called | query={query!r}")

    detector, data, crops_path = _get_detector_and_data()
    logger.info(f"[{trace_id}] crops_path={crops_path!r} | crops_count={len(data.get('crops', []))}")

    # 1) Local Fuzzy Detection
    local = detector.identify_crop(query, top_k=5)

    best = local.get("best")
    ambiguous = bool(local.get("ambiguous"))
    candidates = local.get("candidates", []) or []

    logger.info(
        f"[{trace_id}] local_result | ambiguous={ambiguous} | best={best} | "
        f"candidates={[c.get('master_name') for c in candidates]}"
    )

    # Curated ambiguity list should override local exact matches (e.g., "lobiya").
    ambiguous_entry = _find_ambiguous_match(query, data)
    if ambiguous_entry:
        options = ambiguous_entry.get("resolves_to", []) or []
        buttons = ambiguous_entry.get("button_options", []) or []
        logger.info(f"[{trace_id}] DECISION=curated_ambiguous | options={options}")
        return {
            "crop_name": None,
            "is_ambiguous": True,
            "ambiguous_crop_names": options,
            "button_options": buttons,
            "matched_by": "curated_ambiguous",
            "trace_id": trace_id
        }

    if best and not ambiguous:
        logger.info(f"[{trace_id}] DECISION=local_best | crop={best['master_name']!r}")
        hi_name = _get_hindi_name_for_master(best["master_name"], data)
        return {
            "crop_name": best["master_name"],
            "crop_name_hi": hi_name,
            "is_ambiguous": False,
            "matched_by": "local",
            "is_existing_crop": True,
            "trace_id": trace_id
        }

    # 2) Ambiguity Handling (Local)
    if ambiguous:
        options = list(dict.fromkeys([c["master_name"] for c in candidates if "master_name" in c]))
        logger.info(f"[{trace_id}] DECISION=local_ambiguous | options={options}")
        return {
            "crop_name": None,
            "is_ambiguous": True,
            "ambiguous_crop_names": options,
            "matched_by": "local_ambiguous",
            "trace_id": trace_id
        }

    # 3) AI Fallback
    master_names = [c["master_name"] for c in data.get("crops", []) if isinstance(c, dict) and c.get("master_name")]
    logger.info(f"[{trace_id}] fallback=LLM | master_names_count={len(master_names)}")

    ai_result = _ai_detect_crop(query, master_names, trace_id=trace_id)
    logger.info(
        f"[{trace_id}] ai_result={{'status': {ai_result.get('status')!r}, "
        f"'crop_name': {ai_result.get('crop_name')!r}}}"
    )

    if ai_result["status"] == "none":
        logger.info(f"[{trace_id}] DECISION=none")
        return {"crop_name": None, "is_ambiguous": False, "matched_by": "none", "trace_id": trace_id}

    ai_crop = (ai_result.get("crop_name") or "").strip() if isinstance(ai_result.get("crop_name"), str) else None
    if not ai_crop:
        logger.warning(f"[{trace_id}] AI returned status={ai_result.get('status')} but empty crop_name.")
        return {"crop_name": None, "is_ambiguous": False, "matched_by": "none", "trace_id": trace_id}

    # If AI says "new", we append with validation
    if ai_result["status"] == "new":
        added = _add_new_crop_to_file(crops_path, ai_crop, ai_result.get("synonyms"))
        hi_name = _pick_hindi_from_synonyms(ai_result.get("synonyms")) or ai_crop
        logger.info(f"[{trace_id}] DECISION=ai_new | crop={ai_crop!r} | file_added={added}")
        return {
            "crop_name": ai_crop,
            "crop_name_hi": hi_name,
            "is_ambiguous": False,
            "matched_by": "ai_new",
            "is_existing_crop": False if added else True,  # if already existed, treat as existing
            "file_added": added,
            "trace_id": trace_id
        }

    # status == "master" from LLM
    hi_name = _get_hindi_name_for_master(ai_crop, data)
    logger.info(f"[{trace_id}] DECISION=ai_master | crop={ai_crop!r}")
    return {
        "crop_name": ai_crop,
        "crop_name_hi": hi_name,
        "is_ambiguous": False,
        "matched_by": "ai_existing",
        "is_existing_crop": True,
        "trace_id": trace_id
    }

    detector, data, crops_path = _get_detector_and_data()

    # 1. Local Fuzzy Detection
    local = detector.identify_crop(query, top_k=5)
    
    if local.get("best") and not local.get("ambiguous"):
        return {
            "crop_name": local["best"]["master_name"],
            "is_ambiguous": False,
            "matched_by": "local",
            "is_existing_crop": True
        }

    # 2. Ambiguity Handling (Curated or Local)
    if local.get("ambiguous"):
        options = list(dict.fromkeys([c["master_name"] for c in local.get("candidates", [])]))
        return {
            "crop_name": None,
            "is_ambiguous": True,
            "ambiguous_crop_names": options,
            "matched_by": "local_ambiguous"
        }

    # 3. AI Fallback
    master_names = [c["master_name"] for c in data.get("crops", []) if "master_name" in c]
    ai_result = _ai_detect_crop(query, master_names)

    if ai_result["status"] == "none":
        return {"crop_name": None, "is_ambiguous": False, "matched_by": "none"}

    ai_crop = ai_result.get("crop_name")
    
    if ai_result["status"] == "new" and ai_crop:
        _add_new_crop_to_file(crops_path, ai_crop, ai_result.get("synonyms"))
        return {
            "crop_name": ai_crop,
            "is_ambiguous": False,
            "matched_by": "ai_new",
            "is_existing_crop": False
        }

    return {
        "crop_name": ai_crop,
        "is_ambiguous": False,
        "matched_by": "ai_existing",
        "is_existing_crop": True
    }
