import re
from datetime import datetime
from .schema import BUNDESLAENDER, KATEGORIEN

def norm_year(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    m = re.match(r"^\s*(19\d{2}|20\d{2})\s*$", s)
    if not m:
        return None
    y = int(m.group(1))
    this = datetime.now().year
    return str(y) if 1900 <= y <= this else None

def norm_number_plain(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = s.replace("€", "").replace("eur", "").replace("euro", "")
    s = s.replace(" ", "").replace(".", "").replace(",", "")
    return s if s.isdigit() else None

def norm_bundesland(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    v = s.strip().lower()
    for bl in BUNDESLAENDER:
        if bl.lower() == v:
            return bl
    aliases = {
        "nrw": "Nordrhein-Westfalen",
        "saxony": "Sachsen",
        "lower saxony": "Niedersachsen",
        "bw": "Baden-Württemberg",
        "bawü": "Baden-Württemberg", "ba-wü": "Baden-Württemberg",
        "mv": "Mecklenburg-Vorpommern",
        "rheinland pfalz": "Rheinland-Pfalz", "rp": "Rheinland-Pfalz",
        "sa": "Sachsen-Anhalt",
    }
    return aliases.get(v)

def is_valid_kategorie(k: str) -> bool:
    return isinstance(k, str) and k.strip() in KATEGORIEN
