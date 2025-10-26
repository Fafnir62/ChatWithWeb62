# core/questions/branche.py
from ..llm import call_llm

SYSTEM = """
Du erkennst die Branche/Industrie eines Unternehmens aus einer kurzen Projektbeschreibung.

Antworte NUR als JSON:
{"answered": true|false, "value": "KURZE BRANCHENBEZEICHNUNG|not found"}

Regeln:
- Setze answered=true NUR, wenn der Text genügend Hinweise liefert, um die Branche
  mit hoher Wahrscheinlichkeit zu bestimmen (auch implizit).
- Ziehe implizite Signale heran: Produkt/Leistung, Zielkunden, Technologien,
  Regulierung, Orte (Krankenhaus, Baustelle), Lieferkette usw.
- value ist eine GENERISCHE, kurze Branchenbezeichnung (1–4 Wörter), z.B.:
  "IT / Software / KI", "Industrie / Maschinenbau", "Bau / Sanierung",
  "MedTech / Gesundheit", "Energie / Erneuerbare", "Landwirtschaft / Food",
  "Handel / E-Commerce", "Gastronomie", "Bildung", "Logistik / Mobilität",
  "Finanzen", "Tourismus", "Immobilien", "Chemie", "Biotech", "Kreativwirtschaft".
- Kein Satz, keine langen Erklärungen, keine Marken/Produkte, keine Orte.
- Wenn die Beschreibung zu allgemein ist: answered=false, value="not found".

Beispiele:
TEXT: "Wir entwickeln eine KI-gestützte SaaS-Plattform für Produktionsplanung." ->
{"answered": true, "value": "IT / Software / KI"}

TEXT: "Wir modernisieren Mehrfamilienhäuser energetisch und ersetzen Heizungen." ->
{"answered": true, "value": "Bau / Sanierung"}

TEXT: "Wir eröffnen ein Café mit Mittagstisch." ->
{"answered": true, "value": "Gastronomie"}

TEXT: "Wir wollen ein Unternehmen gründen." ->
{"answered": false, "value": "not found"}
"""

def check_branche(text: str) -> dict:
    obj = call_llm(SYSTEM, text or "")
    answered = bool(obj.get("answered"))
    value = obj.get("value")

    if not answered:
        return {"answered": False, "value": "not found"}

    if not isinstance(value, str) or not value.strip():
        return {"answered": False, "value": "not found"}

    # keep it short and tidy
    cleaned = " ".join(value.strip().split())
    # hard cap to ~4-5 words
    cleaned = " ".join(cleaned.split()[:5])

    return {"answered": True, "value": cleaned}
