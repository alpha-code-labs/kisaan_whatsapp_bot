#!/usr/bin/env python3
"""
Standalone Crop Detector (Hindi + English + misspellings + optional transliteration)

Usage:
  1) Put your crop JSON in a file (example: crops.json) with shape:
     { "crops": [ { "master_name": "...", "synonyms": [ {"en":"..","hi":".."}, ... ] }, ... ] }

  2) Run:
     python crop_detector.py --crops crops.json --query "nimbu me keede lag gye hai"

Or import and call:
  from crop_detector import CropDetector
  detector = CropDetector.from_json_file("crops.json", enable_transliteration=True)
  result = detector.identify_crop("नीम्बू में कीड़े लग गए")
"""

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process


# ----------------------------
# Normalization helpers
# ----------------------------

# Keep: letters/digits/underscore/space + Devanagari
_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u097F]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def has_devanagari(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", s))

def has_latin(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


# ----------------------------
# Optional Transliteration Layer
# ----------------------------

class OptionalTransliteration:
    """
    Uses indic-transliteration when available:
      - Devanagari -> Roman (ITRANS + HK)
      - Roman (if detectable as a known scheme) -> Devanagari
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.available = False
        if not enabled:
            return
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            from indic_transliteration import detect
            self.sanscript = sanscript
            self.transliterate = transliterate
            self.detect = detect
            self.available = True
        except Exception:
            self.available = False

    def alias_variants(self, alias_norm: str) -> List[str]:
        """Generate extra alias forms to store in the index."""
        if not (self.enabled and self.available) or not alias_norm:
            return []
        variants: List[str] = []
        try:
            if has_devanagari(alias_norm):
                variants.append(self.transliterate(alias_norm, self.sanscript.DEVANAGARI, self.sanscript.ITRANS).lower())
                variants.append(self.transliterate(alias_norm, self.sanscript.DEVANAGARI, self.sanscript.HK).lower())

            if has_latin(alias_norm):
                scheme = self.detect.detect(alias_norm)
                if scheme:
                    variants.append(self.transliterate(alias_norm, scheme, self.sanscript.DEVANAGARI))
        except Exception:
            return []

        # Unique
        out, seen = [], set()
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def query_variants(self, q_norm: str) -> List[str]:
        """Generate extra query forms to try matching on."""
        if not (self.enabled and self.available) or not q_norm:
            return []
        variants: List[str] = []
        try:
            if has_devanagari(q_norm):
                variants.append(self.transliterate(q_norm, self.sanscript.DEVANAGARI, self.sanscript.ITRANS).lower())
                variants.append(self.transliterate(q_norm, self.sanscript.DEVANAGARI, self.sanscript.HK).lower())

            if has_latin(q_norm):
                scheme = self.detect.detect(q_norm)
                if scheme:
                    variants.append(self.transliterate(q_norm, scheme, self.sanscript.DEVANAGARI))
        except Exception:
            return []

        out, seen = [], set()
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out


# ----------------------------
# Match structures
# ----------------------------

@dataclass
class MatchResult:
    master_name: str
    score: float              # 0..100
    match_type: str           # exact_word | exact_phrase | fuzzy_token | fuzzy_phrase
    matched_alias: str


# ----------------------------
# CropDetector
# ----------------------------

class CropDetector:
    def __init__(self, crops: List[Dict[str, Any]], enable_transliteration: bool = True):
        self.crops = crops
        self.xlit = OptionalTransliteration(enabled=enable_transliteration)

        # alias -> list of master_names (collisions possible!)
        self.alias_to_masters: Dict[str, List[str]] = {}

        # master_name -> set(aliases)
        self.master_to_aliases: Dict[str, set] = {}

        self.single_word_aliases: List[str] = []
        self.multi_word_aliases: List[str] = []

        self._build_index()

    @classmethod
    def from_json_file(cls, path: str, enable_transliteration: bool = True) -> "CropDetector":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        crops = data["crops"] if isinstance(data, dict) and "crops" in data else data
        return cls(crops=crops, enable_transliteration=enable_transliteration)

    def _store_alias(self, master: str, alias_norm: str) -> None:
        self.alias_to_masters.setdefault(alias_norm, [])
        if master not in self.alias_to_masters[alias_norm]:
            self.alias_to_masters[alias_norm].append(master)
        self.master_to_aliases.setdefault(master, set()).add(alias_norm)

    def _add_alias(self, master: str, alias_raw: str) -> None:
        alias = normalize_text(alias_raw)
        if not alias:
            return

        # store original normalized alias
        self._store_alias(master, alias)

        # store transliteration variants (optional)
        for v in self.xlit.alias_variants(alias):
            v_norm = normalize_text(v)
            if v_norm and v_norm != alias:
                self._store_alias(master, v_norm)

    def _build_index(self) -> None:
        for item in self.crops:
            master = item["master_name"]

            # also include master_name itself
            self._add_alias(master, master)

            for syn in item.get("synonyms", []):
                en = syn.get("en", "")
                hi = syn.get("hi", "")
                if en:
                    self._add_alias(master, en)
                if hi:
                    self._add_alias(master, hi)

        all_aliases = list(self.alias_to_masters.keys())
        for a in all_aliases:
            (self.multi_word_aliases if " " in a else self.single_word_aliases).append(a)

        # Prefer longer multi-word matches first (more specific)
        self.multi_word_aliases.sort(key=len, reverse=True)

    # ----------------------------
    # Public API
    # ----------------------------

    def identify_crop(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Returns:
          {
            "best": {master_name, score, match_type, matched_alias} | None,
            "candidates": [...],
            "ambiguous": bool
          }
        """
        q_norm = normalize_text(query)
        if not q_norm:
            return {"best": None, "candidates": [], "ambiguous": False}

        # Build query variants (transliteration adds extra forms)
        variants = [q_norm] + [normalize_text(v) for v in self.xlit.query_variants(q_norm)]
        # unique keep order
        variants = [v for v in dict.fromkeys(variants) if v]

        all_candidates: List[MatchResult] = []
        for qv in variants:
            all_candidates.extend(self._detect_single_variant(qv))

            # If we got exact hits on this variant, we can optionally stop early
            # (comment this out if you want maximum recall)
            if any(c.match_type.startswith("exact") for c in all_candidates):
                break

        ranked = self._rank_candidates(all_candidates)
        return self._finalize(ranked, top_k=top_k)

    # ----------------------------
    # Internal detection per variant
    # ----------------------------

    def _detect_single_variant(self, q_norm: str) -> List[MatchResult]:
        candidates: List[MatchResult] = []
        q_tokens = q_norm.split()
        token_set = set(q_tokens)

        # 1) Exact phrase match (multi-word aliases)
        for alias in self.multi_word_aliases:
            if alias in q_norm:
                for master in self.alias_to_masters[alias]:
                    candidates.append(MatchResult(master, 100.0, "exact_phrase", alias))

        # 2) Exact word match (single-word aliases)
        for alias in self.single_word_aliases:
            if alias in token_set:
                for master in self.alias_to_masters[alias]:
                    candidates.append(MatchResult(master, 100.0, "exact_word", alias))

        if candidates:
            return candidates

        # 3) Fuzzy token match for misspellings
        for tok in q_tokens:
            if len(tok) < 2:
                continue

            # dynamic threshold (stricter for short tokens)
            if len(tok) <= 4:
                threshold = 92
            elif len(tok) <= 7:
                threshold = 88
            else:
                threshold = 85

            m = process.extractOne(tok, self.single_word_aliases, scorer=fuzz.ratio)
            if not m:
                continue
            alias, score, _ = m
            if score >= threshold:
                for master in self.alias_to_masters[alias]:
                    candidates.append(MatchResult(master, float(score), "fuzzy_token", alias))

        # 4) Fuzzy phrase match (alias as part of query)
        if len(q_norm) >= 4 and self.multi_word_aliases:
            m = process.extractOne(q_norm, self.multi_word_aliases, scorer=fuzz.partial_ratio)
            if m:
                alias, score, _ = m
                if score >= 88:
                    for master in self.alias_to_masters[alias]:
                        candidates.append(MatchResult(master, float(score), "fuzzy_phrase", alias))

        return candidates

    def _rank_candidates(self, candidates: List[MatchResult]) -> List[MatchResult]:
        if not candidates:
            return []

        type_priority = {
            "exact_phrase": 4,
            "exact_word": 3,
            "fuzzy_phrase": 2,
            "fuzzy_token": 1,
        }

        best_by_master: Dict[str, MatchResult] = {}
        for c in candidates:
            prev = best_by_master.get(c.master_name)
            if not prev:
                best_by_master[c.master_name] = c
                continue

            if (c.score > prev.score) or (
                c.score == prev.score
                and type_priority.get(c.match_type, 0) > type_priority.get(prev.match_type, 0)
            ):
                best_by_master[c.master_name] = c

        ranked = list(best_by_master.values())
        ranked.sort(key=lambda x: (x.score, type_priority.get(x.match_type, 0)), reverse=True)
        return ranked

    def _finalize(self, ranked: List[MatchResult], top_k: int) -> Dict[str, Any]:
        if not ranked:
            return {"best": None, "candidates": [], "ambiguous": False}

        # If top-2 are very close, mark ambiguous (important for your Lemon vs Acid Lime “nimbu/नींबू”)
        ambiguous = False
        if len(ranked) >= 2 and (ranked[0].score - ranked[1].score) <= 2:
            ambiguous = True

        best = None if ambiguous else self._as_dict(ranked[0])

        return {
            "best": best,
            "candidates": [self._as_dict(r) for r in ranked[:top_k]],
            "ambiguous": ambiguous
        }

    @staticmethod
    def _as_dict(r: MatchResult) -> Dict[str, Any]:
        return {
            "master_name": r.master_name,
            "score": round(r.score, 2),
            "match_type": r.match_type,
            "matched_alias": r.matched_alias,
        }


# ----------------------------
# CLI entrypoint
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Crop Detector (Hindi/English + fuzzy + optional transliteration)")
    parser.add_argument("--crops", required=True, help="Path to crops.json (either {crops:[...]} or just [...])")
    parser.add_argument("--query", required=True, help="Farmer query string")
    parser.add_argument("--no-xlit", action="store_true", help="Disable transliteration layer")
    parser.add_argument("--topk", type=int, default=5, help="How many candidates to return")
    args = parser.parse_args()

    detector = CropDetector.from_json_file(args.crops, enable_transliteration=(not args.no_xlit))
    result = detector.identify_crop(args.query, top_k=args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    