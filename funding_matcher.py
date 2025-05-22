# funding_matcher.py
import json, os
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_DATA_FILE = Path("foerdermittel_samples.json")
_INDEX_DIR = "funding_index"


# ──────────────────────────
# Load dataset as list[dict]
# ──────────────────────────
def _load_dataset() -> List[Dict]:
    with _DATA_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    items: List[Dict] = []
    for cat, lst in raw.items():
        for prog in lst:
            items.append({**prog, "category": cat})
    return items


# ──────────────────────────
# Build / load FAISS index
# ──────────────────────────
def build_or_load_index() -> FAISS:
    emb = OpenAIEmbeddings()
    if os.path.isdir(_INDEX_DIR):
        return FAISS.load_local(_INDEX_DIR, emb, allow_dangerous_deserialization=True)

    docs: List[Document] = []
    for p in _load_dataset():
        text = f"{p['title']}\n{p['description']}\n{' '.join(p.get('keywords', []))}"
        docs.append(Document(page_content=text, metadata=p))

    index = FAISS.from_documents(docs, emb)
    index.save_local(_INDEX_DIR)
    return index


# ──────────────────────────
# API 1  – keep old "top-n"
# ──────────────────────────
def top_n_matches(user_profile: str, n: int = 3) -> List[Dict]:
    """
    Return the n most similar programmes (lower score = better).
    """
    index = build_or_load_index()
    hits: List[Tuple[Document, float]] = index.similarity_search_with_score(user_profile, k=n)
    return [doc.metadata | {"score": score} for doc, score in hits]


# ──────────────────────────
# API 2  – score threshold
# ──────────────────────────
def matches_above_threshold(
    user_profile: str,
    min_score: float = 0.30,
    k: int = 20
) -> List[Dict]:
    """
    Return ALL programmes whose similarity score ≤ min_score.
    • min_score ≈ 0.25–0.30 for strict matches, 0.35+ for looser.
    • k = how many nearest neighbours to fetch before filtering.
    """
    index = build_or_load_index()
    hits: List[Tuple[Document, float]] = index.similarity_search_with_score(user_profile, k=k)
    qualified = [
        doc.metadata | {"score": score}
        for doc, score in hits
        if score <= min_score
    ]
    return qualified
