# foerderparser/extract_funding_category.py

from .utils import clean_spaces, ai_complete_json_object

INNOVATION_KW = [
    "innov", "forsch", "entwicklung", "f&e", "f & e",
    "prototyp", "technologie", "digitalis", "ki ", "künstliche intelligenz",
    "innovation", "innovationsprojekt", "neue produkte", "produktentwicklung",
    "forschungsprojekt",
]

INVESTITION_KW = [
    "invest", "investition", "anschaffung", "ausstattung",
    "anlage", "anlagen", "maschine", "maschinen",
    "modernisierung", "sanierung", "umbau", "ausbau", "erweiterung",
    "bau ", "gebäude", "gebäudesanierung", "infrastruktur", "produktionserweiterung",
]

FINANZIERUNG_KW = [
    "beteiligung", "beteiligungskapital", "garantie", "ausfallgarantie",
    "bürgschaft", "liquidität", "unternehmensfinanzierung", "finanzierung",
    "kredit", "darlehen", "haftungsfreistellung", "sicherheiten",
]

def _keyword_hit(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(kw in low for kw in keywords)

def _ai_classify_category(text: str) -> str:
    """
    Ask AI to classify into:
    - "Innovation"
    - "Investition"
    - "Finanzierung"
    - OR "not found" if it really can't decide based on funding purpose.
    """
    system_prompt = (
        "Du bist ein Klassifizierer für Förderprogramme.\n"
        "Gib ein JSON-Objekt zurück: "
        "{\"funding_category\": \"Innovation\"|\"Investition\"|\"Finanzierung\"|\"not found\"}.\n"
        "\n"
        "Definitionen:\n"
        "- Innovation: Förderung von Forschung & Entwicklung, neuen Technologien, neuen Produkten, Digitalisierung, Prototypen.\n"
        "- Investition: Förderung von physischen/materiellen Investitionen (Maschinen, Anlagen, Bau, Modernisierung, Erweiterung, Infrastruktur).\n"
        "- Finanzierung: Kapitalbereitstellung zur Unternehmensfinanzierung oder Liquidität, Beteiligungskapital, Garantien, Bürgschaften, Kredite, Darlehen.\n"
        "- not found: Wenn der Text zu allgemein ist oder du keine klare Zuordnung machen kannst.\n"
        "\n"
        "Wähle GENAU EINEN Wert."
    )

    user_prompt = (
        "Ordne dieses Förderprogramm einer Kategorie zu.\n\n"
        f"{text}\n\n"
        "Antwort NUR als JSON wie {\"funding_category\": \"Investition\"}."
    )

    obj = ai_complete_json_object(system_prompt, user_prompt)

    if not obj:
        return "not found"

    cat_raw = obj.get("funding_category")
    if not isinstance(cat_raw, str):
        return "not found"

    cat = cat_raw.strip()
    if cat in ("Innovation", "Investition", "Finanzierung", "not found"):
        return cat

    return "not found"

def extract_funding_category(detail1: str, detail2: str) -> str:
    """
    Decide which top-level bucket this program belongs to.

    Priority:
    1. Rule-based keyword match:
       - Innovation
       - Investition
       - Finanzierung
    2. If still unclear: ask AI. AI may return "not found".
    """

    full = clean_spaces(detail1 + " " + detail2)

    # Heuristic priority:
    if _keyword_hit(full, INNOVATION_KW):
        return "Innovation"

    if _keyword_hit(full, INVESTITION_KW):
        return "Investition"

    if _keyword_hit(full, FINANZIERUNG_KW):
        return "Finanzierung"

    # Fallback: ask AI (can return "not found")
    return _ai_classify_category(full)
