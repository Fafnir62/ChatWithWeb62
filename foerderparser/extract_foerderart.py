# foerderparser/extract_foerderart.py

import re
from typing import List
from .utils import clean_spaces, FOERDERART_TARGETS, ai_complete_json_object

def _extract_after_label(label: str, blob: str) -> str:
    """
    Extracts the text that comes after e.g. 'Förderart:' in Detail1,
    until the next '  Something:' block or end.
    """
    if not blob:
        return ""
    regex = rf"{label}\s*(.+?)(?:\s{2,}[A-ZÄÖÜa-zäöü]+:|$)"
    m = re.search(regex, blob, flags=re.I | re.S)
    if not m:
        return ""
    return clean_spaces(m.group(1))

def _normalize_token(token: str) -> str:
    """
    Map a raw word like 'Garantie' or 'Kredit' to one of:
    'Bürgschaften', 'Zuschuss', 'Darlehen'.
    Returns '' if we can't map.
    """
    t = token.lower().strip(" .,:;()")
    return FOERDERART_TARGETS.get(t, "")

def _scan_text_for_types(text: str) -> List[str]:
    """
    Look for any substrings in FOERDERART_TARGETS anywhere in the text,
    and collect mapped canonical types.
    """
    found: List[str] = []
    low = text.lower()
    for raw_kw, mapped in FOERDERART_TARGETS.items():
        if raw_kw in low:
            if mapped not in found:
                found.append(mapped)
    return found

def _ai_guess_foerderart(fulltext: str) -> List[str]:
    """
    Ask AI to classify the Förderart.
    Output must be one or more of: 'Bürgschaften', 'Zuschuss', 'Darlehen'.
    If multiple apply, include multiple in the array.
    We'll reject anything outside that set.
    """
    system_prompt = (
        "Du bist ein Klassifizierer für Förderprogramme.\n"
        "Gib ein JSON-Objekt zurück wie z.B. {\"förderart\": [\"Zuschuss\"]}.\n\n"
        "Regeln:\n"
        "- 'Zuschuss': nicht rückzahlbare finanzielle Unterstützung / Zuschuss / Zuwendung.\n"
        "- 'Darlehen': rückzahlbares Geld / Kredit / Darlehen / zinsgünstiges Darlehen.\n"
        "- 'Bürgschaften': Garantien, Ausfallgarantien, Haftungsfreistellungen, Bürgschaften.\n"
        "Mehrere Einträge sind erlaubt, aber NUR diese drei Werte.\n"
        "Wenn unklar, nimm den wahrscheinlichsten.\n"
    )

    user_prompt = (
        "Bestimme die Förderart aus diesem Text.\n\n"
        f"{fulltext}\n\n"
        "Ergebnis bitte als JSON mit dem Feld 'förderart', z.B. {\"förderart\": [\"Zuschuss\"]}."
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)

    if not obj:
        return []

    val = obj.get("förderart")
    if not isinstance(val, list):
        return []

    cleaned: List[str] = []
    for item in val:
        if not isinstance(item, str):
            continue
        canon = item.strip()
        if canon in ("Bürgschaften", "Zuschuss", "Darlehen"):
            if canon not in cleaned:
                cleaned.append(canon)

    return cleaned

def extract_foerderart(detail1: str, detail2: str) -> List[str]:
    """
    Final public function.
    Steps:
    1. Try to extract from 'Förderart:' block in Detail1.
    2. If nothing, scan both Detail1 + Detail2 for known keywords.
    3. If still nothing, ask AI to classify.
    4. If STILL nothing, return ['not found'].
    """
    d1 = detail1 or ""
    d2 = detail2 or ""
    fulltext = (d1 + "\n\n" + d2).strip()

    collected: List[str] = []

    # Step 1: direct 'Förderart:' field in Detail1
    raw = _extract_after_label("Förderart:", d1)
    if raw:
        parts = re.split(r"[,/]| und | bzw\.? ", raw, flags=re.I)
        for p in parts:
            norm = _normalize_token(p)
            if norm and norm not in collected:
                collected.append(norm)

    # Step 2: keyword scan in full text if still empty
    if not collected:
        collected = _scan_text_for_types(fulltext)

    # Step 3: AI fallback if still empty
    if not collected and fulltext:
        collected = _ai_guess_foerderart(fulltext)

    # Step 4: final fallback
    if not collected:
        return ["not found"]

    return collected
