# core/questions/eigenanteil.py
import re
from ..llm import call_llm
from ..validators import norm_number_plain

# ── Hard gate: require Eigenmittel markers in free text (when no question context) ──
_EIGEN_MARKERS = re.compile(
    r"(eigenmittel|eigen\-?anteil|eigenkapital|aus eigenen mitteln|"
    r"\bwir bringen\b|\bwir steuern\b|\bwir können\b|\bwir koennen\b|\bwir leisten\b|"
    r"\bintern bereit\b|\bselbst finanzieren\b|\bselbst tragen\b)",
    flags=re.I
)
_CURRENCY_MARKERS = re.compile(r"(€|\beur\b|\beuro\b|\bmio\b|million|milliarde|\btsd\b|tausend)", re.I)

def _has_eigen_markers(txt: str) -> bool:
    t = txt or ""
    # must mention eigen-idea explicitly; currency alone is NOT enough
    return bool(_EIGEN_MARKERS.search(t))

SYSTEM = """
Extrahiere den **Eigenanteil / Eigenmittel** (in EUR) des Unternehmens.

ANTWORTE NUR ALS JSON:
{"answered": true|false, "value": "NUMBER|not found"}

KONTEXTREGEL:
- Wenn **keine FRAGE** vorliegt (nur freie Beschreibung):
  - Antworte **nur dann** mit answered=true, wenn der Betrag eindeutig als
    Eigenmittel/Eigenanteil bezeichnet ist (z. B. "Eigenmittel", "Eigenanteil",
    "aus eigenen Mitteln", "wir bringen X mit", "wir steuern X bei", "wir können X bereitstellen").
  - Ein **nackter Zahlenwert** (z. B. "5000") oder nur Währungsangabe **ohne** Eigen-Kontext → answered=false.
- Wenn die FRAGE explizit nach dem Eigenanteil fragt und die ANTWORT eine Zahl ist,
  interpretiere sie als Eigenanteil in EUR.

IGNORIEREN:
- Projektkosten/Gesamtbudget, Zuschuss/Fördersumme, Kredit/Darlehen/Bürgschaft, reine Prozentangaben.

Regeln:
1) Gib **nur Ziffern** zurück (z. B. "300000").
2) Bei Bereichen ("200.000–300.000 € Eigenmittel"): nimm die **untere Grenze**.
3) Unklar → {"answered": false, "value":"not found"}.
4) Nur JSON ausgeben.

Beispiele:
FRAGE: Wie viel Eigenanteil können Sie aufbringen?
ANTWORT/BEZ: 5000
-> {"answered": true, "value": "5000"}

FRAGE: —
ANTWORT/BEZ: Eigenmittel: 0,5 Mio.
-> {"answered": true, "value": "500000"}

FRAGE: —
ANTWORT/BEZ: 5000
-> {"answered": false, "value": "not found"}

FRAGE: —
ANTWORT/BEZ: Wir können 150 Tsd. € aus eigenen Mitteln tragen.
-> {"answered": true, "value": "150000"}
"""

def check_eigenanteil(text: str, context: str | None = None) -> dict:
    # HARD GATE for free text: require Eigenmittel markers (currency alone is not enough)
    if not context and not _has_eigen_markers(text or ""):
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

    # Normalize to digits only (handles "300.000", "0,5 Mio.", etc.)
    n = norm_number_plain(value)
    if n:
        return {"answered": True, "value": n}
    return {"answered": False, "value": "not found"}
