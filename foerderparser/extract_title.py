# foerderparser/extract_title.py
from typing import Tuple
from .utils import clean_spaces, ai_complete_json_object

def extract_title(titel_raw: str, detail1: str, detail2: str) -> str:
    titel_raw = clean_spaces(titel_raw)
    d1 = clean_spaces(detail1)
    d2 = clean_spaces(detail2)

    # Rule 1: direct title
    if titel_raw:
        return titel_raw

    # Rule 2: generate with AI if we have detail1 or detail2
    if d1 or d2:
        system_prompt = (
            "Du bist ein Assistent, der Förderprogramme benennt.\n"
            "Gib ein JSON-Objekt zurück: {\"title\": \"...\"}.\n"
            "title: kurzer offizieller Programmtitel in Deutsch (max 12 Wörter).\n"
            "Kein Fluff, keine Einleitung."
        )
        user_prompt = (
            f"Erzeuge einen Programmtitel für folgendes Förderprogramm.\n\n"
            f"DETAIL1:\n{d1}\n\nDETAIL2:\n{d2}\n"
        )
        obj = ai_complete_json_object(system_prompt, user_prompt)
        if obj and isinstance(obj.get("title"), str) and obj["title"].strip():
            return obj["title"].strip()

    # Rule 3: nothing available
    return "not found"
