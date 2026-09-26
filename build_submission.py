"""Build the required 30-line static artifact from the canonical generated test set."""
import json
from pathlib import Path
from bot import compose

ROOT = Path(__file__).parent
DATA = ROOT / "dataset"

def load_folder(name, key):
    return {data[key]: data for path in (DATA / name).glob("*.json")
            if (data := json.loads(path.read_text(encoding="utf-8")))}

categories = load_folder("categories", "slug")
merchants = load_folder("merchants", "merchant_id")
customers = load_folder("customers", "customer_id")
triggers = load_folder("triggers", "id")
pairs = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]

with (ROOT / "submission.jsonl").open("w", encoding="utf-8") as out:
    for pair in pairs:
        merchant = merchants[pair["merchant_id"]]
        message = compose(categories[merchant["category_slug"]], merchant,
                          triggers[pair["trigger_id"]], customers.get(pair.get("customer_id")))
        out.write(json.dumps({"test_id": pair["test_id"], **message}, ensure_ascii=False) + "\n")

print("Wrote 30 context-grounded messages to submission.jsonl")
