# core/questions/kategorie.py
from ..llm import call_llm
from ..validators import is_valid_kategorie

SYSTEM = """
Du ordnest eine kurze Projektbeschreibung genau EINER Kategorie zu:

- Innovation: Forschung & Entwicklung, Prototypen, neue/weiterentwickelte Technologien o. Produkte,
  KI/ML, Digitalisierung als Entwicklung (nicht nur Einführung).
  Signalwörter: entwickeln, F&E, Prototyp, Pilot, neue Technologie, Algorithmus, Plattform (neu).

- Investition: Materielle/immaterielle Anschaffungen, Ausbau, Modernisierung, Umbau,
  energetische Sanierung/Energieeffizienz (Dach-/Fassadendämmung, Heizung, PV),
  Maschinen/Anlagen, Gebäude, Infrastruktur.
  Signalwörter: modernisieren, sanieren, umbauen, erweitern, anschaffen, investieren, Halle, Maschine,
  Dämmung, Heizung, energetisch, PV, Gebäude, Produktionslinie.

- Finanzierung: Kapital-/Liquiditätsbedarf, Kredit, Darlehen, Bürgschaft, Garantie,
  Beteiligungskapital, Haftungsfreistellung – OHNE konkrete Investitions-/Sanierungsmaßnahmen.

Entscheidungsregeln bei Überschneidung:
1) Wenn echte F&E/Neu-Entwicklung klar vorliegt → Innovation.
2) Sonst, wenn Investitionen/Modernisierung/Sanierung/Anschaffung im Vordergrund → Investition.
3) Sonst, wenn es primär um Kapitalinstrumente geht → Finanzierung.
Wenn unklar: not found.

Antworte NUR als JSON:
{"answered": true|false, "value": "Innovation|Investition|Finanzierung|not found"}

Beispiele:
TEXT: "Wir entwickeln eine KI-gestützte Analysesoftware (Prototyp) und testen mit Pilotkunden."
-> {"answered": true, "value": "Innovation"}

TEXT: "Wir modernisieren unsere Produktionshalle energetisch (Dach- und Fassadendämmung, neue Heizung)."
-> {"answered": true, "value": "Investition"}

TEXT: "Wir benötigen Betriebsmittelkredit / Bürgschaft zur Liquiditätssicherung."
-> {"answered": true, "value": "Finanzierung"}

TEXT: "Wir planen ein Projekt und suchen Fördermittel."
-> {"answered": false, "value": "not found"}
"""

def check_kategorie(text: str) -> dict:
    obj = call_llm(SYSTEM, text or "")
    answered = bool(obj.get("answered"))
    value = obj.get("value", "not found")

    if answered and is_valid_kategorie(value):
        return {"answered": True, "value": value}
    # if model returned an invalid label, treat as not found
    return {"answered": False, "value": "not found"}
