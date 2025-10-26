# foerderparser/extract_hoehe.py

import re
from typing import Optional
from .utils import clean_spaces, ai_complete_json_object

# Regex candidates:
# 1. Max/bis ... Euro amounts with currency markers
# 2. % coverage with funding-ish context

AMOUNT_PATTERNS = [
    # "maximal 1,25 Mio. EUR", "max. EUR 1.250.000", "bis zu 500.000 Euro"
    r"(max(?:imal|\.?)\s[^.\n]{0,60}?(?:EUR|€|Euro)[^.\n]{0,60})",
    r"(bis zu\s[^.\n]{0,60}?(?:EUR|€|Euro)[^.\n]{0,60})",
    r"(bis\s[^.\n]{0,60}?(?:EUR|€|Euro)[^.\n]{0,60})",
    r"(\d{1,3}(?:\.\d{3})*(?:,\d+)?\s*(?:EUR|€|Euro)[^.\n]{0,40})",
    r"(\d+(?:,\d+)?\s*Mio\.?\s*(?:EUR|€|Euro)[^.\n]{0,40})",
    r"(\d+(?:,\d+)?\s*Million(?:en)?\s*(?:EUR|€|Euro)[^.\n]{0,40})",
]

PERCENT_PATTERNS = [
    # "Die Garantie deckt bis zu 75 % der Beteiligungssumme"
    # "Haftungsfreistellung von 60 Prozent der Kreditsumme"
    r"(bis zu\s*\d{1,3}\s*%[^.\n]{0,80}?(Beteiligung|Finanzierung|Kosten|Kredit|Kreditsumme|Risiko|Förderung|Investition))",
    r"(\d{1,3}\s*%[^.\n]{0,80}?(Beteiligung|Finanzierung|Kosten|Kredit|Kreditsumme|Risiko|Förderung|Investition))",
    r"(\d{1,3}\s*Prozent[^.\n]{0,80}?(Beteiligung|Finanzierung|Kosten|Kredit|Kreditsumme|Risiko|Förderung|Investition))",
]

def _find_candidate_phrase(detail2: str) -> Optional[str]:
    """
    Try to pull a plausible 'Höhe der Förderung' phrase using regex.
    Only accepts phrases that clearly include currency or % in a funding context.
    Returns the FIRST match that looks valid.
    """

    txt = clean_spaces(detail2)

    # First try explicit Euro amount patterns
    for patt in AMOUNT_PATTERNS:
        m = re.search(patt, txt, flags=re.I)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate

    # Then try percentage coverage
    for patt in PERCENT_PATTERNS:
        m = re.search(patt, txt, flags=re.I)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate

    return None

def _ai_validate_candidate(candidate: str, fulltext: str) -> bool:
    """
    Ask AI: does this candidate actually describe Förderhöhe / Förderanteil,
    or is it something unrelated (like '140 Millionen Personen')?

    Returns True if AI says "Ja, das ist die Förderhöhe", else False.
    """

    system_prompt = (
        "Du bekommst einen kurzen Ausschnitt (candidate) und den vollen Kontext (text).\n"
        "Beantworte als JSON: {\"is_foerderhoehe\": true/false}.\n\n"
        "true NUR WENN der Ausschnitt beschreibt:\n"
        "- maximale Fördersumme / Obergrenze in EUR/Euro/€\n"
        "- Anteil der Förderung in Prozent (z.B. 75% der Kosten werden übernommen)\n"
        "- Bürgschafts-/Garantieanteil in Prozent\n"
        "NICHT true, wenn es um Anzahl Personen, Reichweite, Laufzeit, Jahre, etc. geht."
    )

    user_prompt = (
        f"candidate:\n{candidate}\n\n"
        f"text:\n{fulltext}\n\n"
        "Liefere NUR JSON wie {\"is_foerderhoehe\": true} oder {\"is_foerderhoehe\": false}."
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)
    if not obj:
        return False

    val = obj.get("is_foerderhoehe")
    return bool(val is True)

def _ai_summarize_amount(detail2: str) -> str:
    """
    Ask AI to produce a short human-readable summary of the Förderhöhe /
    Deckelung / Anteil. Max ~20 Wörter.
    """

    text = clean_spaces(detail2)
    if not text:
        return ""

    system_prompt = (
        "Du liest die Beschreibung eines Förderprogramms und fasst NUR die Förderhöhe "
        "(max. Betrag, Prozentsatz der Kosten, Deckelungen) in einem kurzen deutschen Satzfragment zusammen.\n"
        "Gib ein JSON zurück wie {\"hoehe_der_foerderung\": \"...\"}.\n"
        "Wenn es keine feste Grenze gibt, sag z.B. 'Höhe nach Einzelfall, keine feste Obergrenze'.\n"
        "Maximal ~20 Wörter."
    )

    user_prompt = (
        "Extrahiere bitte die Förderhöhe / Fördersumme / Anteil aus diesem Text:\n\n"
        f"{text}\n\n"
        "Antwort NUR als JSON mit Schlüssel 'hoehe_der_foerderung'."
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)
    if not obj:
        return ""

    val = obj.get("hoehe_der_foerderung")
    if not isinstance(val, str):
        return ""

    result = val.strip()
    if not result:
        return ""

    # safety: don't let it ramble paragraphs
    if len(result.split()) > 30:
        return ""

    return result

def extract_hoehe(detail2: str) -> str:
    """
    Final logic for 'höhe_der_förderung':

    1. Run regex to find a candidate phrase with Euro or %.
    2. Ask AI if this candidate is actually about Förderhöhe. If yes, return candidate.
    3. If no usable candidate, ask AI to summarize funding amount/limits from context.
    4. If AI can't extract anything meaningful, return 'not found'.
    """

    fulltext = clean_spaces(detail2)

    # Step 1: find candidate via regex
    candidate = _find_candidate_phrase(detail2)

    # Step 2: validate candidate with AI (guard against '140 Millionen Menschen')
    if candidate and fulltext:
        if _ai_validate_candidate(candidate, fulltext):
            return candidate

    # Step 3: fallback — ask AI to summarize from scratch
    ai_guess = _ai_summarize_amount(detail2)
    if ai_guess:
        return ai_guess

    # Step 4: nothing found
    return "not found"
