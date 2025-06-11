# augment_dataset.py – enrich a *flat* programme list
# ---------------------------------------------------
"""
Input  : foerdermittel_raw.json                      (list[dict] with keys
          ─ "Titel", "Detail1", "Detail2")
Output : foerdermittel_enriched.json                 (dict[str,list] for the app)

Each enriched object now contains

    title                 ← copy of "Titel"
    description           ← short 1-sentence summary
    funding_area          ← "Bund" | "Land"  (derived from "Fördergebiet")
    call_id               ← best effort (string or "–")
    submission_deadline   ← ISO date "YYYY-MM-DD" or "laufend" or "–"
    förderart             ← one of ["Zuschuss", "Darlehen", "Garantien"]
    höhe_der_förderung    ← short string or "–"
    category              ← one of the six business categories below
    alldetails            ← the full, untouched Detail1 + Detail2 string

Business categories
-------------------
- Forschung und Entwicklung
- Sanierung (insbesondere Energiesanierung)
- Existenzgründung- & Festigung
- Erneuerbare Energien
- Anlagenbau
- Maschinenfinanzierung

Prerequisites
-------------
    • OPENAI_API_KEY in your environment (see .env)
    • pip install openai>=1.21 python-dotenv tqdm
"""

from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

RAW_FILE = Path("foerdermittel_raw.json")
OUT_FILE = Path("foerdermittel_enriched.json")
MODEL    = "gpt-4o-mini"

# ---------------------------------------------------------------------------

load_dotenv()
client = OpenAI()
if not client.api_key:
    sys.exit("❌  Please set OPENAI_API_KEY (e.g. in .env)")

# ----- constant helpers ----------------------------------------------------

_rx_förderart   = re.compile(r"Förderart:\s*([^\n]+)", re.I)
_rx_gebiet      = re.compile(r"Fördergebiet:\s*([^\n]+)", re.I)
_rx_deadline    = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})|(\d{4}-\d{2}-\d{2})", re.I)
_rx_amount      = re.compile(r"(bis .*?€|max[^.]{0,80}€)", re.I)

_NORMALISE_FÖRDERART = {
    "zuschuss":  "Zuschuss",
    "darlehen":  "Darlehen",
    "bürgschaft": "Garantien",
    "garantie":   "Garantien",
}

_CATEGORIES: Dict[str, List[str]] = {
    "Forschung und Entwicklung"               : ["forschung", "innovation"],
    "Sanierung (insbesondere Energiesanierung)": ["sanierung", "modernisierung", "energ"],
    "Existenzgründung- & Festigung"            : ["gründ", "existenz"],
    "Erneuerbare Energien"                     : ["solar", "wind", "energie", "erneuerbar"],
    "Anlagenbau"                               : ["anlage", "anlagen"],
    "Maschinenfinanzierung"                    : ["maschine", "maschinen"],
}

def auto_category(text: str) -> str:
    lower = text.lower()
    for cat, keys in _CATEGORIES.items():
        if any(k in lower for k in keys):
            return cat
    return "Sonstiges"

# ----- Chat-GPT prompt -----------------------------------------------------

SYSTEM_PROMPT = """
Du bist Datenanalyst für öffentliche Förder­programme.
Liefere **genau ein** JSON-Objekt (kein Markdown, keine Code-Fence).

Felder & Regeln
---------------
title               : Unverändert aus TITLE.
description         : 1–3 **deutsche** Sätze (≤ 55 Wörter) – Worum geht es? Für wen ist es gedacht?
funding_area        : "Bund" bei bundesweiter Gültigkeit, sonst "Land".
call_id             : Aktenzeichen, sonst "–".
submission_deadline : • Erstes Datum → ISO "YYYY-MM-DD"
                      • Wörter wie "laufend" → "laufend"
                      • sonst "–".
förderart           : **Liste** mit genau einem Eintrag aus ["Zuschuss", "Darlehen", "Garantien"].
                      ➜ Immer aus `Detail1` entnehmen: steht direkt nach „Förderart: …“.
höhe_der_förderung  : kurzer Betrag oder Satz, z. B. "bis 500 000 €", "zwischen 40 und 80 %", sonst "–".

Format
------
• Alle Felder genau einmal und in dieser Reihenfolge.
• Alle Felder als Strings, außer 'förderart': Liste mit genau 1 String.
• Keine Code-Fences, keine Zeilenumbrüche innerhalb der Werte.
• Nur ein einzelnes JSON-Objekt zurückgeben.
"""

# ---------------------------------------------------------------------------

def chat_extract(title: str, detail1: str, detail2: str) -> dict:
    """Let GPT create the structured skeleton. Regex will patch leftovers."""
    desc_for_llm = (detail1 + "\n\n" + detail2)[:3500]  # stay under token limit
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"TITLE: {title}\n\nDESCRIPTION:\n{desc_for_llm}"},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        # unbelievably bad answer – fall back to an empty shell
        return {
            "title":                title,
            "description":          "",
            "funding_area":         "–",
            "call_id":              "–",
            "submission_deadline":  "–",
            "förderart":            "–",
            "höhe_der_förderung":   "–",
        }
    
GERMAN_STATES_AND_CITIES = [
    # Bundesländer (states)
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
    "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen",
    "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein",
    "Thüringen",

    # Major cities
    "München", "Berlin", "Hamburg", "Köln", "Frankfurt", "Stuttgart", "Düsseldorf",
    "Dortmund", "Essen", "Leipzig", "Bremen", "Dresden", "Hannover", "Nürnberg",
    "Duisburg", "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster", "Karlsruhe",
    "Mannheim", "Augsburg", "Wiesbaden", "Gelsenkirchen", "Mönchengladbach",
    "Braunschweig", "Chemnitz", "Kiel", "Aachen", "Halle", "Magdeburg", "Freiburg",
    "Krefeld", "Lübeck", "Mainz", "Erfurt", "Oberhausen", "Rostock", "Kassel", "Hagen",
    "Saarbrücken", "Potsdam", "Oldenburg", "Würzburg", "Heidelberg", "Regensburg",
    "Wolfsburg", "Ulm", "Ingolstadt", "Heilbronn", "Pforzheim", "Göttingen", "Reutlingen",
    "Trier", "Jena", "Siegen", "Gera", "Cottbus", "Zwickau", "Koblenz", "Hildesheim",
    "Witten", "Flensburg", "Gütersloh", "Konstanz", "Dessau-Roßlau", "Schwerin"
]

def infer_funding_area_from_locations(text: str) -> str:
    lower_text = text.lower()
    # Check for specific state mentions first
    for location in GERMAN_STATES_AND_CITIES:
        if location.lower() in lower_text:
            return location
    # Fallback for clearly federal-wide mentions
    if "bundesweit" in lower_text or "ganz deutschland" in lower_text:
        return "Bund"
    return "Land"
# ---------------------------------------------------------------------------

def regex_patch(record: dict, detail1: str, detail2: str) -> dict:
    """Fill missing or obviously wrong fields via regex heuristics."""
    full = f"{detail1}\n{detail2}"

    # funding_area  (Bund / Land <X>)
    # Always re-parse funding_area if GPT guessed "Land"
    if record["funding_area"] in ("–", "", None, "Land"):
        m = _rx_gebiet.search(full)
        if m:
            record["funding_area"] = infer_funding_area_from_locations(m.group(1))
        else:
            record["funding_area"] = infer_funding_area_from_locations(full)
    
    if record["funding_area"] != "Land":
        print(f"[✔] Adjusted funding_area to: {record['funding_area']}")



    # förderart
    fa = record.get("förderart", ["–"])
    if not isinstance(fa, list) or fa[0] not in ("Zuschuss", "Darlehen", "Garantien"):
        m = _rx_förderart.search(full)
        if m:
            norm = _NORMALISE_FÖRDERART.get(m.group(1).strip().lower(), "–")
            if norm in ("Zuschuss", "Darlehen", "Garantien"):
                record["förderart"] = [norm]
            else:
                record["förderart"] = ["–"]

    # submission_deadline
    if record["submission_deadline"] in ("–", "", None):
        m = _rx_deadline.search(full)
        if m:
            d = m.group(0).replace(".", "-")
            record["submission_deadline"] = (
                "-".join(reversed(d.split("-"))) if d.count(".") else d
            )

    # höhe_der_förderung
    if record["höhe_der_förderung"] in ("–", "", None):
        m = _rx_amount.search(full)
        if m:
            record["höhe_der_förderung"] = m.group(0).strip()

    return record


def main() -> None:
    if not RAW_FILE.exists():
        sys.exit(f"❌  {RAW_FILE} not found")

    raw: List[Dict] = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    enriched: Dict[str, List] = {}

    for prog in tqdm(raw, desc="items"):
        title   = prog.get("Titel", "").strip()
        detail1 = prog.get("Detail1", "").strip()
        detail2 = prog.get("Detail2", "").strip()

        rec = chat_extract(title, detail1, detail2)
        rec = regex_patch(rec, detail1, detail2)

        rec["alldetails"] = f"{detail1}\n\n{detail2}".strip()
        rec["category"]   = auto_category(rec["description"] + detail1 + detail2)

        cat_bucket = rec["category"]
        enriched.setdefault(cat_bucket, []).append(rec)

        time.sleep(1.2)        # keep a *very* safe distance to the 60-req/min limit

    OUT_FILE.write_text(json.dumps(enriched, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"✅  {sum(len(v) for v in enriched.values())} items → {OUT_FILE}")


if __name__ == "__main__":
    main()
