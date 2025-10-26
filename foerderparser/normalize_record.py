# foerderparser/normalize_record.py

from typing import Dict, Any

from .extract_title import extract_title
from .extract_description import extract_description
from .extract_funding_area import extract_funding_area
from .extract_funding_category import extract_funding_category
from .extract_foerderart import extract_foerderart
from .extract_hoehe import extract_hoehe
from .extract_alldetails import extract_alldetails
from .utils import clean_spaces


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn one scraped record into our final normalized format.
    """

    titel_raw = raw.get("Titel", "") or ""
    detail1   = raw.get("Detail1", "") or ""
    detail2   = raw.get("Detail2", "") or ""

    # Run individual extractors
    out_title            = extract_title(titel_raw, detail1, detail2)
    out_description      = extract_description(detail1, detail2)
    out_funding_area     = extract_funding_area(detail1)
    out_funding_category = extract_funding_category(detail1, detail2)
    out_foerderart       = extract_foerderart(detail1, detail2)
    out_hoehe            = extract_hoehe(detail2)
    out_alldetails       = extract_alldetails(detail1, detail2)

    # Fallbacks / cleanup ---------------------------------

    # title
    if not out_title or not out_title.strip():
        out_title = "not found"

    # description
    if not out_description or not out_description.strip():
        out_description = "not found"

    # funding_area (default to Bund if nothing we can detect)
    if not out_funding_area or not out_funding_area.strip():
        out_funding_area = "Bund"

    # funding_category
    if out_funding_category not in ("Innovation", "Investition", "Finanzierung", "not found"):
        out_funding_category = "not found"

    # förderart should always be list
    if not isinstance(out_foerderart, list) or len(out_foerderart) == 0:
        out_foerderart = ["not found"]

    # höhe_der_förderung
    if not out_hoehe or not out_hoehe.strip():
        out_hoehe = "not found"

    # alldetails
    if not out_alldetails or not out_alldetails.strip():
        out_alldetails = "not found"

    # Produce final normalized object
    return {
        "title": clean_spaces(out_title),
        "description": clean_spaces(out_description),
        "funding_area": clean_spaces(out_funding_area),
        "funding_category": clean_spaces(out_funding_category),
        "förderart": out_foerderart,  # keep list
        "höhe_der_förderung": clean_spaces(out_hoehe),
        "alldetails": out_alldetails.strip(),
    }
