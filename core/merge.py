from .schema import REQUIRED_FIELDS

def merge_answers(current: dict, updates: dict) -> dict:
    out = current.copy()
    for k in out.keys():
        v = (updates or {}).get(k, "not found")
        if v not in (None, "", "not found"):
            out[k] = v
    return out

def pretty_status(ans: dict) -> str:
    lines = []
    for key, label in REQUIRED_FIELDS:
        val = ans.get(key, "not found")
        if key == "gruendungsjahr" and ans.get("kategorie") != "Innovation":
            # Not required unless Innovation
            if val in (None, "", "not found"):
                val = "– (nur bei Innovation erforderlich)"
        lines.append(f"- **{label}**: {val}")
    return "\n".join(lines)
