# foerderparser/utils.py
import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_spaces(s: str) -> str:
    if not s:
        return ""
    # collapse all whitespace into single spaces
    return re.sub(r"\s+", " ", s).strip()

GERMAN_STATES = [
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
    "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen",
    "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein",
    "Thüringen",
]

# Förderart normalization target set
FOERDERART_TARGETS = {
    "garantie": "Bürgschaften",
    "garantien": "Bürgschaften",
    "ausfallgarantie": "Bürgschaften",
    "bürgschaft": "Bürgschaften",
    "bürgschaften": "Bürgschaften",

    "zuschuss": "Zuschuss",
    "zuschüsse": "Zuschuss",
    "förderzuschuss": "Zuschuss",
    "zuwendung": "Zuschuss",
    "zuwendungen": "Zuschuss",

    "darlehen": "Darlehen",
    "darlehn": "Darlehen",
    "kredit": "Darlehen",
    "kredite": "Darlehen",
    "nachrangdarlehen": "Darlehen",
    "finanzierungskredit": "Darlehen",
}

def ai_complete_json_object(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> Optional[dict]:
    """
    Helper to call the model for structured small tasks.
    Returns parsed JSON or None if something went wrong.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return None
