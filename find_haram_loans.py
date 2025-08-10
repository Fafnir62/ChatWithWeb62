import json, os, re, sys, time
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

RAW_FILE = Path("foerdermittel_raw.json")
OUT_FILE = Path("foerdermittel_enriched.json")
HARAM_TITLES_FILE = Path("haram_titles.txt")
MODEL    = "gpt-4o-mini"

load_dotenv()
client = OpenAI()
if not client.api_key:
    sys.exit("❌ Please set OPENAI_API_KEY (e.g. in .env)")

# Regex patterns
_rx_förderart = re.compile(r"Förderart:\s*([^\n]+)", re.I)
_rx_gebiet    = re.compile(r"Fördergebiet:\s*([^\n]+)", re.I)
_rx_deadline  = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})|(\d{4}-\d{2}-\d{2})", re.I)
_rx_amount    = re.compile(r"(bis .*?€|max[^.]{0,80}€)", re.I)

_NORMALISE_FÖRDERART = {
    "zuschuss": "Zuschuss",
    "darlehen": "Darlehen",
    "bürgschaft": "Garantien",
    "garantie": "Garantien",
}

SYSTEM_PROMPT = """
Du bist Datenanalyst für öffentliche Förderprogramme.
Liefere **genau ein** JSON-Objekt (kein Markdown).

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
    """Ask OpenAI to normalize one record"""
    desc_for_llm = (detail1 + "\n\n" + detail2)[:3500]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"TITLE: {title}\n\nDESCRIPTION:\n{desc_for_llm}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {
            "title": title,
            "description": "",
            "funding_area": "–",
            "call_id": "–",
            "submission_deadline": "–",
            "förderart": ["–"],
            "höhe_der_förderung": "–",
        }

def detect_haram(full_text: str, foerderart: List[str]) -> str:
    """Detect if a program is a Darlehen with interest (haram)"""
    text = full_text.lower()
    has_darlehen = "darlehen" in text or (foerderart and "Darlehen" in foerderart)
    has_interest = any(w in text for w in ["zins", "zinsen", "zinstragend", "zinsgünstig"])
    if has_darlehen and has_interest:
        return "haram"
    elif has_darlehen:
        return "maybe"
    return "ok"

def main():
    if not RAW_FILE.exists():
        sys.exit(f"❌ {RAW_FILE} not found")

    raw = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    enriched: List[dict] = []
    haram_titles: List[str] = []

    for prog in tqdm(raw, desc="Analysiere Programme"):
        title = prog.get("Titel", "").strip()
        detail1 = prog.get("Detail1", "").strip()
        detail2 = prog.get("Detail2", "").strip()

        rec = chat_extract(title, detail1, detail2)

        full_text = f"{rec['title']} {rec['description']} {detail1} {detail2}"
        risk = detect_haram(full_text, rec.get("förderart", []))
        rec["haram_risk"] = risk
        rec["alldetails"] = f"{detail1}\n\n{detail2}".strip()

        enriched.append(rec)
        if risk == "haram":
            haram_titles.append(rec["title"])

        time.sleep(1.2)  # avoid rate limit

    # Write full enriched dataset
    OUT_FILE.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write only haram titles
    HARAM_TITLES_FILE.write_text("\n".join(haram_titles), encoding="utf-8")

    print(f"\n✅ Analyse abgeschlossen.")
    print(f"Alle Daten: {OUT_FILE}")
    print(f"Haram-Darlehen Titel: {HARAM_TITLES_FILE}")
    print("\nGefundene Programme mit Darlehen + Zinsen:")
    for t in haram_titles:
        print(" -", t)

if __name__ == "__main__":
    main()
