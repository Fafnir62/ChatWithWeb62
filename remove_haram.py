import json
from pathlib import Path

# Paths
input_file = Path(r"C:\Users\Hai-m\OneDrive\Desktop\FördermittelV\ChatWithWeb62\foerdermittel_enriched2.json")
output_file = input_file.with_name("foerdermittel_enriched.json")

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter out entries with haram_risk == "haram"
filtered_data = [entry for entry in data if entry.get("haram_risk") != "haram"]

# Save the filtered list back to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved filtered data to: {output_file}")
print(f"Removed {len(data) - len(filtered_data)} entries with haram_risk = 'haram'")
