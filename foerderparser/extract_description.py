# foerderparser/extract_description.py
from .utils import clean_spaces, ai_complete_json_object

def extract_description(detail1: str, detail2: str) -> str:
    d1 = clean_spaces(detail1)
    d2 = clean_spaces(detail2)

    if not d1 and not d2:
        return "not found"

    text_for_ai = (d1 + "\n\n" + d2).strip()

    system_prompt = (
        "Du bist ein Assistent für öffentliche Förderprogramme.\n"
        "Du fasst die Förderung in 2-3 Sätzen auf Deutsch zusammen "
        "(max 70 Wörter insgesamt). Du sollst klar erklären:\n"
        "- Wer kann sie bekommen?\n"
        "- Wofür ist sie gedacht?\n"
        "- Wie unterstützt sie (z.B. Zuschuss, Garantie, Darlehen)?\n"
        "Antwort nur als JSON {\"description\": \"...\"}."
    )

    user_prompt = (
        "Erzeuge eine verständliche Kurzbeschreibung.\n\n"
        f"Alle Details:\n{text_for_ai}\n"
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)
    if obj and isinstance(obj.get("description"), str) and obj["description"].strip():
        return obj["description"].strip()

    # fallback: just return first ~250 chars
    fallback = text_for_ai[:250]
    return fallback if fallback else "not found"
