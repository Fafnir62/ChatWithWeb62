# matching.py this one runs on mathematical comparison, and does the filtering before that, but no ai here
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Basic German tokenization + tiny stopword list (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

GERMAN_STOPWORDS = {
    "und","oder","der","die","das","ein","eine","einer","einem","einen","den","dem",
    "zu","mit","für","im","in","am","an","auf","aus","als","bei","vom","von","des",
    "ist","sind","werden","wird","auch","sowie","bis","dass","daß","nach","vor",
    "durch","ohne","unter","über","so","wenn","diese","dieser","dieses","denn","etc",
    "sie","er","es","wir","ihr","ihnen","ihre","ihren","ihrer","euch","man","kann",
    "können","nicht","nur","noch","schon"
}

UMLAUT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
    "ß": "ss"
})

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.translate(UMLAUT_MAP).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    norm = normalize_text(text)
    tokens = [t for t in norm.split(" ") if t and t not in GERMAN_STOPWORDS and not t.isdigit()]
    return tokens

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight BM25 (no external libraries)
# ─────────────────────────────────────────────────────────────────────────────

def bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]], k1=1.5, b=0.75) -> List[float]:
    N = len(docs_tokens)
    # doc length + avg len
    doc_lens = [len(d) for d in docs_tokens]
    avgdl = sum(doc_lens) / N if N else 0.0

    # DF for query terms
    df = defaultdict(int)
    query_set = set(query_tokens)
    for dtoks in docs_tokens:
        unique = set(dtoks)
        for qt in query_set:
            if qt in unique:
                df[qt] += 1

    # IDF
    idf = {}
    for qt in query_set:
        # standard BM25 idf with +1 to keep positive
        idf_val = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
        idf[qt] = idf_val

    scores = []
    for idx, dtoks in enumerate(docs_tokens):
        tf = Counter(dtoks)
        denom = k1 * (1 - b + b * (doc_lens[idx] / (avgdl or 1.0)))  # guard div0
        s = 0.0
        for qt in query_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            s += idf[qt] * ((f * (k1 + 1)) / (f + denom))
        scores.append(s)
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

BUNDESLAND_SYNONYMS = {
    "nrw": "Nordrhein-Westfalen",
    "nordrhein westfalen": "Nordrhein-Westfalen",
    "baden wuerttemberg": "Baden-Württemberg",
    "baden-wuerttemberg": "Baden-Württemberg",
    "bayern": "Bayern",
    "berlin": "Berlin",
    "brandenburg": "Brandenburg",
    "bremen": "Bremen",
    "hamburg": "Hamburg",
    "hessen": "Hessen",
    "mecklenburg vorpommern": "Mecklenburg-Vorpommern",
    "niedersachsen": "Niedersachsen",
    "rheinland pfalz": "Rheinland-Pfalz",
    "saarland": "Saarland",
    "sachsen": "Sachsen",
    "sachsen anhalt": "Sachsen-Anhalt",
    "schleswig holstein": "Schleswig-Holstein",
    "thueringen": "Thüringen",
    "thüringen": "Thüringen",
}

VALID_KATEGORIEN = {"Innovation", "Investition", "Finanzierung"}

def normalize_bundesland(bl: str) -> str:
    if not bl or bl == "not found":
        return ""
    key = normalize_text(bl)
    key = key.replace("-", " ").strip()
    return BUNDESLAND_SYNONYMS.get(key, bl)

def parse_number_eur(val) -> float:
    """
    Best-effort: convert a value like "0", "0.0", 0, "10000", "10.000", "10 000"
    to float. Unknown -> -1
    """
    if val is None or val == "not found":
        return -1.0
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return -1.0
    s = str(val)
    s = s.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return -1.0

def _has_zuschuss(foerderart_list: List[str]) -> bool:
    if not foerderart_list:
        return False
    return any("zuschuss" in (fa or "").lower() for fa in foerderart_list)

# ─────────────────────────────────────────────────────────────────────────────
# Matching main
# ─────────────────────────────────────────────────────────────────────────────

def load_programs(json_path: str) -> List[Dict]:
    if not os.path.isabs(json_path):
        base = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base, json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def apply_filters(programs: List[Dict], answers: Dict) -> List[Dict]:
    """
    Filters:
      1) Kategorie (Innovation, Investition, Finanzierung)
      2) Bundesland: keep only items with funding_area == BL OR "Bund"
      3) Zuschuss filter: if eigenanteil == 0 => drop Zuschuss Fördermittel
    """
    kategorie = answers.get("kategorie", "")
    bl = normalize_bundesland(answers.get("bundesland", ""))
    eigen = parse_number_eur(answers.get("eigenanteil_eur"))

    filtered = []

    for p in programs:
        # (1) Kategorie
        cat = (p.get("funding_category") or "").strip()
        if kategorie in VALID_KATEGORIEN:
            if cat != kategorie:
                continue

        # (2) Bundesland (or Bund)
        area = (p.get("funding_area") or "").strip()
        if bl:
            if not (area == bl or area == "Bund"):
                continue

        # (3) Zuschuss filter if eigenanteil is 0
        foerderart = p.get("förderart") or []
        if eigen == 0.0 and _has_zuschuss(foerderart):
            continue

        filtered.append(p)

    return filtered

def build_doc_text(p: Dict) -> str:
    parts = [
        p.get("title", ""),
        p.get("description", ""),
        p.get("alldetails", ""),
        p.get("funding_category", ""),
        p.get("funding_area", ""),
        " ".join(p.get("förderart") or []),
        p.get("höhe_der_förderung", ""),
    ]
    return " \n".join([str(x) for x in parts if x])

def match_programs(
    answers: Dict,
    project_text: str,
    json_path: str = "foerdermittel_normalized.json",
    top_k: int = 3
) -> List[Dict]:
    """
    Returns top_k matched programs after filtering, scored via BM25
    over title+description+alldetails against the user's project_text.
    """
    programs = load_programs(json_path)
    filtered = apply_filters(programs, answers)

    if not filtered:
        return []

    # Query tokens from the user's free text (can be the first message or the latest, up to the app)
    query_tokens = tokenize(project_text or "")

    # If user text is empty/very short, fall back to category & branch as query context
    if len(query_tokens) < 3:
        add_bits = []
        if answers.get("branche") and answers["branche"] != "not found":
            add_bits.append(str(answers["branche"]))
        if answers.get("kategorie") and answers["kategorie"] in VALID_KATEGORIEN:
            add_bits.append(str(answers["kategorie"]))
        query_tokens = tokenize(" ".join(add_bits)) or ["foerderung"]

    # Prepare docs
    docs_text = [build_doc_text(p) for p in filtered]
    docs_tokens = [tokenize(t) for t in docs_text]

    # Score via BM25
    scores = bm25_scores(query_tokens, docs_tokens)

    ranked = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
    top = [r[0] for r in ranked[:top_k]]
    return top

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit rendering helper (kept separate so app.py stays very small)
# ─────────────────────────────────────────────────────────────────────────────

def render_matches(matches: List[Dict]):
    """
    Call this from Streamlit to show the Top-N results nicely.
    """
    try:
        import streamlit as st
    except Exception:
        # If used outside Streamlit, just print
        for i, m in enumerate(matches, 1):
            print(f"{i}. {m.get('title','(ohne Titel)')}")
        return

    if not matches:
        st.info("Kein Treffer nach den Filtern. Ändere eine Angabe oder beschreibe dein Projekt ausführlicher.")
        return

    for i, m in enumerate(matches, 1):
        st.markdown(f"### {i}. {m.get('title','(ohne Titel)')}")
        meta_bits = []
        area = m.get("funding_area")
        if area:
            meta_bits.append(f"**Fördergebiet:** {area}")
        cat = m.get("funding_category")
        if cat:
            meta_bits.append(f"**Kategorie:** {cat}")
        foerderart = ", ".join([fa for fa in (m.get('förderart') or []) if fa and fa != 'not found'])
        if foerderart:
            meta_bits.append(f"**Förderart:** {foerderart}")
        amount = m.get("höhe_der_förderung")
        if amount:
            meta_bits.append(f"**Höhe der Förderung:** {amount}")

        if meta_bits:
            st.markdown(" • ".join(meta_bits))

        if m.get("description"):
            st.markdown(m["description"])

        details = m.get("alldetails")
        if details:
            with st.expander("Alle Details"):
                st.markdown(details.replace("\n\n", "\n\n"))
        st.divider()
