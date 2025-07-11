import json, os, re, sys, time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

RAW_FILE = Path("foerdermittel_raw.json")
OUT_FILE = Path("foerdermittel_enriched.json")
MODEL    = "gpt-4o-mini"

load_dotenv()
client = OpenAI()
if not client.api_key:
    sys.exit("❌  Please set OPENAI_API_KEY (e.g. in .env)")

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
    "Forschung und Entwicklung"                : ["forschung", "innovation"],
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

SYSTEM_PROMPT = """
Du bist Datenanalyst für öffentliche Förder­programme.
Liefere **genau ein** JSON-Objekt (kein Markdown, keine Code-Fence).

Felder & Regeln
---------------
title               : Unverändert aus TITLE.
description         : 1–3 **deutsche** Sätze (≤ 55 Wörter).
funding_area        : "Bund" bei bundesweiter Gültigkeit, sonst "Land".
call_id             : Aktenzeichen, sonst "–".
submission_deadline : Datum ISO "YYYY-MM-DD" oder "laufend" oder "–".
förderart           : Liste mit genau einem Eintrag aus ["Zuschuss", "Darlehen", "Garantien"].
höhe_der_förderung  : kurzer Satz oder Betrag.
"""

def chat_extract(title: str, detail1: str, detail2: str) -> dict:
    desc_for_llm = (detail1 + "\n\n" + detail2)[:3500]
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
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
    "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen",
    "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein",
    "Thüringen",
]

def infer_funding_area_from_locations(text: str) -> str:
    lower_text = text.lower()
    for location in GERMAN_STATES_AND_CITIES:
        if location.lower() in lower_text:
            return location
    if "bundesweit" in lower_text or "ganz deutschland" in lower_text:
        return "Bund"
    return "Land"

# ---------- NEW: log-friendly parser for Detail1 ----------------
def extract_foerderbereich_from_detail1(detail1: str) -> str | None:
    matches = re.findall(r"Förderbereich:\s*(.*?)\s*(?:Förder|$)", detail1, flags=re.I | re.S)
    if matches:
        result = matches[0].strip()
        print(f"[LOG] Extracted Förderbereich: {result}")
        return result
    print("[LOG] No Förderbereich found in Detail1.")
    return None

def regex_patch(record: dict, detail1: str, detail2: str) -> dict:
    full = f"{detail1}\n{detail2}"

    if record["funding_area"] in ("–", "", None, "Land"):
        m = _rx_gebiet.search(full)
        if m:
            record["funding_area"] = infer_funding_area_from_locations(m.group(1))
        else:
            record["funding_area"] = infer_funding_area_from_locations(full)
    if record["funding_area"] != "Land":
        print(f"[✔] Adjusted funding_area to: {record['funding_area']}")

    fa = record.get("förderart", ["–"])
    if not isinstance(fa, list) or fa[0] not in ("Zuschuss", "Darlehen", "Garantien"):
        m = _rx_förderart.search(full)
        if m:
            norm = _NORMALISE_FÖRDERART.get(m.group(1).strip().lower(), "–")
            record["förderart"] = [norm if norm in ("Zuschuss", "Darlehen", "Garantien") else "–"]

    if record["submission_deadline"] in ("–", "", None):
        m = _rx_deadline.search(full)
        if m:
            d = m.group(0).replace(".", "-")
            record["submission_deadline"] = "-".join(reversed(d.split("-"))) if d.count(".") else d

    if record["höhe_der_förderung"] in ("–", "", None):
        m = _rx_amount.search(full)
        if m:
            record["höhe_der_förderung"] = m.group(0).strip()

    # NEW LOGIC: Extract Förderbereich
    fb_value = extract_foerderbereich_from_detail1(detail1)
    if fb_value:
        print(f"[LOG] Raw Förderbereich value: {fb_value}")
        parts = [part.strip() for part in fb_value.split(",")]
        for part in parts:
            cat = auto_category(part)
            print(f"[LOG] Testing part '{part}' -> category '{cat}'")
            if cat != "Sonstiges":
                print(f"[✔] FINAL category from Förderbereich: {cat}")
                record["category"] = cat
                break
        else:
            if not record.get("category") or record["category"] == "Sonstiges":
                record["category"] = auto_category(detail1 + detail2)
    else:
        if not record.get("category") or record["category"] == "Sonstiges":
            record["category"] = auto_category(detail1 + detail2)

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

        print(f"\n[== ITEM ==] {title}")
        print(f"[DETAIL1]: {detail1}")

        rec = chat_extract(title, detail1, detail2)
        rec = regex_patch(rec, detail1, detail2)

        rec["alldetails"] = f"{detail1}\n\n{detail2}".strip()
        
        if not rec.get("category") or rec["category"] == "Sonstiges":
            rec["category"] = auto_category(rec["description"] + detail1 + detail2)

        print(f"[RESULT] category: {rec['category']}")

        cat_bucket = rec["category"]
        enriched.setdefault(cat_bucket, []).append(rec)

        time.sleep(1.2)

    OUT_FILE.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅  {sum(len(v) for v in enriched.values())} items → {OUT_FILE}")

if __name__ == "__main__":
    main()
