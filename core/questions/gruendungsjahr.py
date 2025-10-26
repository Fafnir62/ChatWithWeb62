# core/questions/gruendungsjahr.py
from ..llm import call_llm
from ..validators import norm_year

SYSTEM = """
Du extrahierst das **Gründungsjahr des Unternehmens** aus einer kurzen Beschreibung.

Antworte NUR als JSON:
{"answered": true|false, "value": "YYYY|not found"}

WICHTIG:
- Gesucht ist ausschließlich das **Gründungsjahr des Unternehmens** (Firmengründung).
- KEINE Projektzeiträume, Förderperioden, Budgetjahre, Baujahre, Zieljahre, Laufzeiten oder Deadlines.
- Typische Hinweise: "gegründet", "Gründung", "besteht seit", "seit <Jahr>", "Firmengründung".
- Beispiele, die **NICHT** das Gründungsjahr sind:
  - "Projekt 2023–2025", "Förderperiode 2021–2027", "bis 2030", "seit 10 Jahren aktiv"
  - "Modernisierung 2019", "Invest in 2022", "Antrag 2024"
- Jahreszahl muss vierstellig (1900–heute) sein. Keine Bereiche (z.B. "2014–2016") und keine Zukunftsjahre.

Regeln:
1) Wenn eine Formulierung die Firmengründung klar benennt (z.B. "gegründet 2018", "besteht seit 2012"), gib dieses Jahr zurück.
2) Wenn mehrere Jahre im Text vorkommen, wähle das Jahr, das **eindeutig** zur Gründung gehört.
3) Unscharfe Angaben wie "seit über 10 Jahren" oder "vor 5 Jahren" → **not found**.
4) Wenn unklar oder keine eindeutige Gründungs-Angabe vorhanden → {"answered": false, "value": "not found"}.
5) Antworte ausschließlich im JSON-Format wie oben.

Beispiele:
TEXT: "Gegründet 2019 in München." -> {"answered": true, "value": "2019"}
TEXT: "Unser Unternehmen besteht seit 2014." -> {"answered": true, "value": "2014"}
TEXT: "Projektlaufzeit 2023–2025; Antrag bis 2024." -> {"answered": false, "value": "not found"}
TEXT: "Wir sind seit über 10 Jahren am Markt." -> {"answered": false, "value": "not found"}
TEXT: "Gründung ca. 2016, seither stetig gewachsen." -> {"answered": true, "value": "2016"}
"""

def check_gruendungsjahr(text: str) -> dict:
    obj = call_llm(SYSTEM, text or "")
    answered = bool(obj.get("answered"))
    value = obj.get("value", "not found")

    # Validate & normalize (four digits, 1900..current year)
    ny = norm_year(value) if isinstance(value, str) else None
    if answered and ny:
        return {"answered": True, "value": ny}

    # Anything else → not found
    return {"answered": False, "value": "not found"}
