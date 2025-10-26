# foerderparser/extract_funding_area.py

import re
from typing import List
from .utils import clean_spaces, GERMAN_STATES, ai_complete_json_object

def _extract_block_after_label(label: str, blob: str) -> str:
    """
    e.g. label='Fördergebiet:' from Detail1.
    We'll grab the text after it until the next double space + NextLabel: or end.
    """
    if not blob:
        return ""
    regex = rf"{label}\s*(.+?)(?:\s{2,}[A-ZÄÖÜa-zäöü]+:|$)"
    m = re.search(regex, blob, flags=re.I | re.S)
    if not m:
        return ""
    return clean_spaces(m.group(1))

def _rule_based_funding_area(block: str) -> str | None:
    """
    Try to classify purely with rules. If confident, return string.
    If not confident, return None so we can ask AI later.
    """

    if not block:
        return None

    block_low = block.lower()

    # 1. bundesweit / deutschland => Bund
    bund_keywords = [
        "bundesweit",
        "deutschland",
        "bundesrepublik",
        "ganz deutschland",
        "bund",
        "federal level",          # just in case english scrape
        "nationwide",
        "germany wide",
        "germany-wide",
        "germanywide",
    ]
    if any(k in block_low for k in bund_keywords):
        return "Bund"

    # 2. detect Bundesländer (could be one or several)
    found_states: List[str] = []
    for land in GERMAN_STATES:
        # match either exact or common abbreviations
        # we'll do a relaxed contains check for now
        if land.lower() in block_low:
            found_states.append(land)

    if found_states:
        # deduplicate but keep order
        seen = []
        for st in found_states:
            if st not in seen:
                seen.append(st)
        return ", ".join(seen)

    # 3. We saw text but couldn't confidently map it.
    # e.g. "Saxony", "NRW", "Meckl.-Vorp."
    # -> return None to trigger AI
    return None

def _ai_guess_funding_area(raw_text: str) -> str:
    """
    Ask AI to interpret Fördergebiet and return either:
    - "Bund"
    - one or more of the EXACT allowed Bundesländer, comma-separated
      Example: "Sachsen" or "Sachsen, Sachsen-Anhalt"

    We'll validate the answer to only allow:
    - "Bund", or
    - items from GERMAN_STATES
    If validation fails, we return "Bund".
    """

    system_prompt = (
        "Du bist ein Klassifizierer für deutsche Förderprogramme.\n"
        "Gib ein JSON-Objekt zurück wie z.B. {\"funding_area\": \"Sachsen\"} oder\n"
        "{\"funding_area\": \"Sachsen, Sachsen-Anhalt\"} oder\n"
        "{\"funding_area\": \"Bund\"}.\n\n"
        "Regeln:\n"
        "- 'Bund' bedeutet: Förderung gilt bundesweit / ganz Deutschland.\n"
        "- Sonst: gib eine durch Komma getrennte Liste von deutschen Bundesländern.\n"
        "- Erlaubte Bundesländer (EXAKT so schreiben):\n"
        "  Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg,\n"
        "  Hessen, Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen,\n"
        "  Rheinland-Pfalz, Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein,\n"
        "  Thüringen\n"
        "- Keine anderen Wörter. Wenn du unsicher bist, nimm 'Bund'."
    )

    user_prompt = (
        "Bestimme das/ die Fördergebiete aus folgendem Text. "
        "Wenn mehrere Länder vorkommen, gib sie kommagetrennt.\n\n"
        f"{raw_text}\n\n"
        "Antworte NUR mit JSON, z.B. {\"funding_area\": \"Nordrhein-Westfalen\"}."
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)

    if not obj:
        return "Bund"

    val = obj.get("funding_area")
    if not isinstance(val, str):
        return "Bund"

    candidate = val.strip()

    # Validate:
    # Option 1: Bund
    if candidate == "Bund":
        return "Bund"

    # Option 2: comma-separated Bundesländer
    parts = [p.strip() for p in candidate.split(",") if p.strip()]
    if not parts:
        return "Bund"

    # keep only valid states, in correct canonical form
    valid_parts: List[str] = []
    for p in parts:
        # We accept only *exact* allowed states
        if p in GERMAN_STATES and p not in valid_parts:
            valid_parts.append(p)

    if not valid_parts:
        # AI returned something weird like "EU-weit" or "NRW, Saxony"
        return "Bund"

    return ", ".join(valid_parts)

def extract_funding_area(detail1: str) -> str:
    """
    Final logic for funding_area:
    1. Try to read Fördergebiet from Detail1 with regex.
    2. Try strict rule-based classification.
    3. If rule-based can't decide confidently (typos, English, abbreviations),
       ask AI to classify.
    4. If AI also fails, default to 'Bund'.
    """

    # Step 1: pull the Fördergebiet block text from Detail1
    block = _extract_block_after_label("Fördergebiet:", detail1)
    block = block.strip()

    # Step 2: try pure rules
    area_rule = _rule_based_funding_area(block)
    if area_rule:
        return area_rule

    # Step 3: AI fallback (only if we have any text to interpret)
    if block:
        return _ai_guess_funding_area(block)

    # Step 4: if there's literally no Fördergebiet info at all
    return "Bund"
