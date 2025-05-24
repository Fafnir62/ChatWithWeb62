# funding_matcher.py

import json, os
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_DATA_FILE = Path("foerdermittel_sample.json")
_INDEX_DIR = "funding_index"

# ──────────────────────────────
# Load dataset as list[dict]
# ──────────────────────────────
def _load_dataset() -> List[Dict]:
    with _DATA_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    items: List[Dict] = []
    for cat, lst in raw.items():
        for prog in lst:
            items.append({**prog, "category": cat})
    return items

# ──────────────────────────────
# Build or load FAISS index
# ──────────────────────────────
def build_or_load_index() -> FAISS:
    emb = OpenAIEmbeddings()
    if os.path.isdir(_INDEX_DIR):
        return FAISS.load_local(_INDEX_DIR, emb, allow_dangerous_deserialization=True)

    docs: List[Document] = []
    for p in _load_dataset():
        text = (
            f"Titel: {p['title']}\n"
            f"Beschreibung: {p['description']}\n"
            f"Zielgruppe: {', '.join(p.get('eligible_applicants', []))}\n"
            f"Fördergebiet: {p.get('funding_area', '')}\n"
            f"Schlagworte: {', '.join(p.get('keywords', []))}\n"
            f"Förderart: {', '.join(p.get('förderart', []))}\n"
            f"Höhe der Förderung: {p.get('höhe_der_förderung', '')}"
        )
        docs.append(Document(page_content=text, metadata=p))

    index = FAISS.from_documents(docs, emb)
    index.save_local(_INDEX_DIR)
    return index

# ──────────────────────────────
# Return top-n matches (lowest scores)
# ──────────────────────────────
def top_n_matches(user_profile: str, n: int = 3) -> List[Dict]:
    index = build_or_load_index()
    hits: List[Tuple[Document, float]] = index.similarity_search_with_score(user_profile, k=n)

    result_fields = [
        "title",
        "description",
        "funding_area",
        "call_id",
        "submission_deadline",
        "förderart",
        "höhe_der_förderung",
        "category"
    ]

    return [
        {key: doc.metadata.get(key) for key in result_fields} | {"score": score}
        for doc, score in hits
    ]

# ──────────────────────────────
# Return matches below a score threshold
# ──────────────────────────────
def matches_above_threshold(
    user_profile: str,
    min_score: float = 0.30,
    k: int = 20
) -> List[Dict]:
    index = build_or_load_index()
    hits: List[Tuple[Document, float]] = index.similarity_search_with_score(user_profile, k=k)

    result_fields = [
        "title",
        "description",
        "funding_area",
        "call_id",
        "submission_deadline",
        "förderart",
        "höhe_der_förderung",
         "category"
    ]

    qualified = []
    for doc, score in hits:
        if score <= min_score:
            filtered = {key: doc.metadata.get(key) for key in result_fields}
            filtered["score"] = score
            qualified.append(filtered)

    return qualified
