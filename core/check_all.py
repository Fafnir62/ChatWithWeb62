# core/check_all.py
from .questions.kategorie import check_kategorie
from .questions.branche import check_branche
from .questions.bundesland import check_bundesland
from .questions.gruendungsjahr import check_gruendungsjahr
from .questions.projektkosten import check_projektkosten
from .questions.eigenanteil import check_eigenanteil

def check_all(text: str) -> dict:
    """
    Run all per-question checkers on the user text and return a partial dict
    with 'not found' where we couldn't extract.
    """
    return {
        "kategorie":         check_kategorie(text).get("value", "not found"),
        "branche":           check_branche(text).get("value", "not found"),
        "bundesland":        check_bundesland(text).get("value", "not found"),
        "gruendungsjahr":    check_gruendungsjahr(text).get("value", "not found"),
        "projektkosten_eur": check_projektkosten(text).get("value", "not found"),
        "eigenanteil_eur":   check_eigenanteil(text).get("value", "not found"),
    }
