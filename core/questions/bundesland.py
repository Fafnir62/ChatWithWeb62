from ..llm import call_llm

SYSTEM = """
Bestimme das **deutsche Bundesland** des Unternehmens.

ANTWORTE NUR ALS JSON:
{"answered": true|false, "value": "EXAKTER_BUNDESLANDNAME|not found"}

Regeln:
- Wenn ein **Bundesland** genannt ist (auch als Abkürzung/englisch/mit Tippfehlern), gib den **exakten deutschen Namen** zurück.
- Wenn eine **Stadt** genannt ist (z. B. "Magdeburg", "München", "Leipzig"), gib das **Bundesland**, in dem die Stadt liegt, zurück.
- Stadtstaaten: "Berlin", "Hamburg", "Bremen" sind **selbst** Bundesländer.
- Bei mehreren Orten wähle den, der am ehesten "Sitz/Standort" bezeichnet.
- Wenn unklar: {"answered": false, "value": "not found"}.
- KEINE Erklärungen, nur JSON.

Beispiele:
TEXT: "Sitz: Magdeburg." -> {"answered": true, "value": "Sachsen-Anhalt"}
TEXT: "Headquartered in Saxony." -> {"answered": true, "value": "Sachsen"}
TEXT: "NRW" -> {"answered": true, "value": "Nordrhein-Westfalen"}
TEXT: "München" -> {"answered": true, "value": "Bayern"}
TEXT: "Halle (Saale)" -> {"answered": true, "value": "Sachsen-Anhalt"}
TEXT: "Berlin" -> {"answered": true, "value": "Berlin"}
TEXT: "—" -> {"answered": false, "value": "not found"}
"""

def check_bundesland(text: str, context: str | None = None) -> dict:
    payload = ""
    if context:
        payload += f"FRAGE: {context}\n"
    payload += f"ANTWORT/BEZ: {text or ''}"
    obj = call_llm(SYSTEM, payload)
    answered = bool(obj.get("answered"))
    value = obj.get("value", "not found")
    if answered and isinstance(value, str) and value.strip():
        return {"answered": True, "value": value.strip()}
    return {"answered": False, "value": "not found"}
