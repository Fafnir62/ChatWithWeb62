# matcher_base.py  – Version mit erweitertem Embedding-Text
from pathlib import Path
import json, os
from typing import List, Dict, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_DATA = Path("foerdermittel_enriched.json")
_INDEX = "funding_index_base"

# -----------------------------------------------------------
def _load() -> List[Dict]:
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    items = []
    for cat, lst in data.items():
        for p in lst:
            items.append({**p, "category": cat})
    return items

# -----------------------------------------------------------
def _make_text(p: Dict) -> str:
    """
    Baut den String, der vektorisiert wird.
    Sie können Felder hinzunehmen oder weglassen.
    """
    parts = [
        p.get("title", ""),
        p.get("description", ""),
        p.get("funding_area", ""),
        ", ".join(p.get("förderart", [])),
        p.get("höhe_der_förderung", ""),
        p.get("category", ""),
        # falls Sie wollen:
        p.get("alldetails", "")[:1000]   # nur ein Auszug, sonst zu lang
    ]
    return "\n".join(filter(None, parts))   # leere Einträge raus

# -----------------------------------------------------------
def _build() -> FAISS:
    emb = OpenAIEmbeddings()
    docs = [
        Document(
            page_content=_make_text(p),
            metadata=p
        ) for p in _load()
    ]
    idx = FAISS.from_documents(docs, emb)
    idx.save_local(_INDEX)
    return idx

def get_index() -> FAISS:
    if os.path.isdir(_INDEX):
        return FAISS.load_local(_INDEX, OpenAIEmbeddings(),
                                allow_dangerous_deserialization=True)
    return _build()

def top_k(profile: str, k: int = 20) -> List[Tuple[Dict, float]]:
    hits = get_index().similarity_search_with_score(profile, k=k)
    return [(doc.metadata, score) for doc, score in hits]
