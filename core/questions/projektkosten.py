# core/questions/projektkosten.py
import re
from ..llm import call_llm

# accept only if at least one of these markers is present in free text (no context)
_COST_MARKERS = re.compile(
    r"(€|\beur\b|\beuro\b|\bmio\b|million|milliarde|\btsd\b|tausend|"
    r"projektkosten|gesamtkosten|budget|volumen|investitionssumme|"
    r"investitionskosten|kosten\b)",
    flags=re.I
)

def _has_cost_markers(txt: str) -> bool:
    return bool(_COST_MARKERS.search(txt or ""))

SYSTEM = """
Extrahiere die **GESAMT-Projektkosten** (Projektvolumen) in EUR.

ANTWORTE NUR ALS JSON:
{"answered": true|false, "value": "NUMBER|not found"}

KONTEXTREGEL:
- Wenn **keine FRAGE** vorliegt (nur freie Beschreibung):
  - Antworte **nur dann** mit answered=true, wenn die Projektkosten **klar bezeichnet** sind
    (z. B. "Projektkosten", "Gesamtkosten", "Budget", "Volumen", "Investitionssumme")
    ODER der Betrag **mit Währung/Einheit** angegeben ist ("€", "EUR", "Euro", "Mio.", "Tsd.").
  - Ein **nackter Zahlenwert** (z. B. "20000") → **answered=false**.
- Wenn eine FRAGE zu Projektkosten vorliegt und die ANTWORT eine Zahl ist,
  interpretiere sie als Projektkosten in EUR.

Definition Projektkosten:
- Gesamtbudget / Projektvolumen / Investitionssumme des Projekts.
IGNORIEREN:
- Eigenanteil/Eigenmittel, Zuschuss/Fördersumme, Kredit/Darlehen/Bürgschaft, Prozentangaben.

Regeln:
1) Bei mehreren Beträgen wähle die **Gesamtsumme**.
2) Bereiche ("800–900 Tsd. €") → **obere Grenze**.
3) Gib **nur Ziffern** zurück (z. B. "1200000").
4) Unklar → {"answered": false, "value": "not found"}.
5) Nur JSON ausgeben.
"""

def check_projektkosten(text: str, context: str | None = None) -> dict:
    # HARD GATE: if this is free text (no question context) and there are no
    # cost markers, we must NOT accept naked numbers.
    if not context and not _has_cost_markers(text or ""):
        return {"answered": False, "value": "not found"}

    payload = ""
    if context:
        payload += f"FRAGE: {context}\n"
    payload += f"ANTWORT/BEZ: {text or ''}"

    obj = call_llm(SYSTEM, payload)
    answered = bool(obj.get("answered"))
    value = obj.get("value", "not found")

    if not answered or not isinstance(value, str) or not value.strip():
        return {"answered": False, "value": "not found"}

    # normalize to digits only
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        return {"answered": True, "value": digits}
    return {"answered": False, "value": "not found"}
