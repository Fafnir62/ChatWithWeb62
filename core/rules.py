from .schema import REQUIRED_FIELDS

def is_applicable(key: str, answers: dict) -> bool:
    if key == "gruendungsjahr":
        return answers.get("kategorie") == "Innovation"
    return True

def is_missing(key: str, answers: dict) -> bool:
    return answers.get(key) in (None, "", "not found")

def next_missing(answers: dict):
    """Return (key, label) of next applicable missing question or (None, None)."""
    for key, label in REQUIRED_FIELDS:
        if not is_applicable(key, answers):
            continue
        if is_missing(key, answers):
            return key, label
    return None, None
