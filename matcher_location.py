# matcher_location.py  ───────  ONLY location logic
import logging, hashlib, pickle
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI

from matcher_base import top_k                           # ← import step-1
_CACHE = Path(".loc_cache.pkl")
_PENALTY = 0.20
_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
_cache: Dict[str, str] = pickle.load(_CACHE.open("rb")) if _CACHE.exists() else {}

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

def normalise(loc: str) -> str:
    if not loc:
        return ""
    q = (f"Gib NUR das Bundesland für '{loc}'. "
         "Ist es bereits ein Bundesland, gib das selbe zurück.")
    return _ask_gpt(q).lower()

# ----------------------------------------------------
def adjusted_matches(profile: str,
                     user_location: str,
                     base_k: int = 20,
                     max_score: float = 0.35) -> List[Dict]:
    """1) get k most similar (step-1)  2) penalise by location"""
    user_state = normalise(user_location)
    out: List[Dict] = []

    for meta, base in top_k(profile, k=base_k):
        prog_state = normalise(meta.get("funding_area", ""))
        score = base
        if prog_state not in ("bund", user_state):
            score += _PENALTY                      # location penalty

        if score <= max_score:
            meta = dict(meta)                      # copy
            meta["score"] = round(score, 3)
            out.append(meta)

    out.sort(key=lambda d: d["score"])
    return out
