# matcher_location.py  ───────  location-aware matching with query compression
import logging
import hashlib
import pickle
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from openai import OpenAI
from matcher_base import top_k  # step-1 similarity (returns List[Tuple[meta, base_score]])

# --------------------------------------------------------------------
# config / cache
_CACHE = Path(".loc_cache.pkl")
_PENALTY = 0.20                 # location mismatch penalty (lower score = better)
_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_cache: Dict[str, str] = pickle.load(_CACHE.open("rb")) if _CACHE.exists() else {}

# --------------------------------------------------------------------
# GPT fallback (used only by normalise when we can't decide fast)
def _ask_gpt(q: str) -> str:
    key = hashlib.sha1(q.encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    client = OpenAI()
    ans = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": q}],
        temperature=0
    ).choices[0].message.content.strip()
    _cache[key] = ans
    pickle.dump(_cache, _CACHE.open("wb"))
    return ans

# --------------------------------------------------------------------
# Location normalisation (kept simple; GPT handles edge cases)
def normalise(loc: str) -> str:
    """
    Return a lower-cased Bundesland name if possible.
    If it's already a Bundesland, returns it. Otherwise asks GPT.
    """
    if not loc:
        return ""
    q = (
        f"Gib NUR das Bundesland für '{loc}'. "
        "Ist es bereits ein Bundesland, gib das selbe zurück."
    )
    return _ask_gpt(q).lower()

# --------------------------------------------------------------------
# Helpers for funding areas (tolerate different separators & 'Bund' synonyms)
_BUND_SYNS = {"bund", "bundesweit", "deutschland", "germany", "federal", "nationwide"}

def _nfkd_lower_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

def _is_bund(s: str) -> bool:
    k = _nfkd_lower_ascii(s).strip()
    return k in _BUND_SYNS

def _split_areas(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[;,/|]", str(raw))
    return [p.strip() for p in parts if p.strip()]

# --------------------------------------------------------------------
# Query compression (to avoid dilution by long/verbose descriptions)
_STOP_DE = {
    "und","oder","aber","mit","für","von","im","in","am","an","auf","der","die","das",
    "den","dem","des","ein","eine","einer","einem","eines","zu","zum","zur","ist","sind",
    "wird","werden","auch","dass","so","sowie","als","bei","per","durch","ohne","nicht",
    "projekt","unternehmen","plattform","tool","lösung","loesung","produkt","system"
}

# Lightweight synonym expansion to improve recall on common domains
_SYNONYM_EXPAND = {
    "ki": {"künstliche intelligenz","kuenstliche intelligenz","ai","ml","machine learning","maschinelles lernen","deep learning"},
    "cybersecurity": {"it-sicherheit","it sicherheit","security","cybersicherheit","informationssicherheit"},
    "forschung": {"f&e","f u e","fue","innovation","innovationsförderung","innovationsfoerderung","zim"},
    "entwicklung": {"produktentwicklung","prototyp","pilot"},
    "cra": {"cyber resilience act"},
    "sbom": {"software bill of materials","bom"},
    "audit": {"compliance","zertifizierung","nachweis"},
}

def _extract_keywords(text: str, top_k: int = 18) -> List[str]:
    t = _nfkd_lower_ascii(text)
    toks = re.findall(r"[a-z0-9\-]+", t)
    toks = [w for w in toks if w not in _STOP_DE and len(w) > 2]
    cnt = Counter(toks)
    return [w for w, _ in cnt.most_common(top_k)]

def _expand_keywords(words: List[str]) -> List[str]:
    out = set(words)
    for w in words:
        key = w.strip().lower()
        if key in _SYNONYM_EXPAND:
            out.update(_SYNONYM_EXPAND[key])
    # normalise missing umlaut variant
    if "kunstliche" in out:
        out.add("künstliche")
    return sorted(out)

def _numbers_hint(text: str) -> str:
    """
    Pull out helpful numeric anchors (years, larger amounts) to bias retrieval.
    """
    t = _nfkd_lower_ascii(text)
    years = re.findall(r"\b(20\d{2})\b", t)
    euros = re.findall(r"\b\d{4,}\b", t)   # rough heuristic: 4+ digits ~ budget/invest
    parts = []
    if years:
        parts.append("jahre:" + ",".join(sorted(set(years))))
    if euros:
        parts.append("betraege:" + ",".join(sorted(set(euros))[:5]))
    return ("hints: " + " ".join(parts)) if parts else ""

def _compress_profile(raw: str) -> str:
    """
    Turn a long, free-form profile into a short, dense query string:
    - keep early header lines (e.g., 'Forschung & Entwicklung')
    - add keywords + synonyms
    - add numeric hints (years/budgets)
    """
    kws = _expand_keywords(_extract_keywords(raw, top_k=18))
    hint = _numbers_hint(raw)
    blocks = []
    if kws:
        blocks.append("keywords: " + ", ".join(kws))
    if hint:
        blocks.append(hint)
    head = "\n".join(raw.splitlines()[:2]).strip()
    if head:
        blocks.insert(0, head)
    return "\n".join(blocks).strip() or raw

# --------------------------------------------------------------------
# Core matching
def adjusted_matches(profile: str,
                     user_location: str,
                     base_k: int = 20,
                     max_score: Optional[float] = 0.35) -> List[Dict]:
    """
    1) Compress long profile into focused keywords (+synonyms, numeric hints)
    2) Retrieve via top_k()
    3) Apply location penalty unless user's state is allowed OR area is Bund/Bundesweit
    4) Filter by threshold if provided (lower = better)
    5) If empty, widen K and relax threshold to still return best candidates
    """
    user_state = normalise(user_location)

    condensed = _compress_profile(profile)

    def _score_entry(meta: Dict, base: float) -> Tuple[Dict, float]:
        score = base
        areas = _split_areas(meta.get("funding_area", ""))
        norm_areas = [normalise(a) for a in areas]
        # penalty only if user state not permitted AND not nationwide
        if user_state and (user_state not in norm_areas) and (not any(_is_bund(a) for a in areas)):
            score += _PENALTY
        m = dict(meta)
        m["score"] = round(score, 3)
        # Clean display: show only the first location label
        if "funding_area" in m:
            m["funding_area"] = areas[0].strip() if areas else ""
        return m, score

    # -------- first pass (strict-ish) --------
    first = top_k(condensed, k=max(base_k, 30))
    scored: List[Dict] = []
    for meta, base in first:
        m, s = _score_entry(meta, base)
        if (max_score is None) or (s <= max_score):
            scored.append(m)

    if scored:
        scored.sort(key=lambda d: d["score"])  # lower score is better
        return scored

    # -------- second pass: wider & no threshold --------
    wider_k = max(base_k * 3, 80)
    second = top_k(condensed, k=wider_k)
    pool: List[Dict] = []
    for meta, base in second:
        m, s = _score_entry(meta, base)
        pool.append(m)

    pool.sort(key=lambda d: d["score"])
    # Return at least base_k, up to 40 best
    return pool[:max(base_k, 40)]
