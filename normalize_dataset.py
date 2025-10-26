# normalize_dataset.py
import json
from pathlib import Path
from foerderparser.normalize_record import normalize_record


def normalize_file(
    input_path: str = "foerdermittel_raw.json",
    output_path: str = "foerdermittel_normalized.json"
):
    """
    Loads all raw records, runs normalization for each,
    and writes the cleaned version to disk as JSON.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {input_path}: {e}")
            return

    print(f"📦 Loaded {len(data)} raw records from {input_path}")

    normalized = []
    for idx, record in enumerate(data, start=1):
        try:
            norm = normalize_record(record)
            normalized.append(norm)
            print(f"✅ [{idx}/{len(data)}] {norm.get('title', 'no title')}")
        except Exception as e:
            print(f"⚠️ Error processing record {idx}: {e}")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"\n✨ Normalization complete! Saved {len(normalized)} entries → {output_file}")


if __name__ == "__main__":
    normalize_file()
