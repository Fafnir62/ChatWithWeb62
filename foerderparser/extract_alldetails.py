# foerderparser/extract_alldetails.py
from .utils import clean_spaces

def extract_alldetails(detail1: str, detail2: str) -> str:
    d1 = detail1.strip() if detail1 else ""
    d2 = detail2.strip() if detail2 else ""
    combo = (d1 + "\n\n" + d2).strip()
    if not combo:
        return "not found"
    # keep structure (line breaks are helpful for debugging),
    # but also collapse crazy inner whitespace in each part
    c1 = clean_spaces(d1)
    c2 = clean_spaces(d2)
    combined = (c1 + "\n\n" + c2).strip() if (c1 or c2) else ""
    return combined if combined else "not found"
