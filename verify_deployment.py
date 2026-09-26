"""Small public endpoint smoke test; uses only synthetic challenge data."""
import json
import sys
from pathlib import Path
from urllib import request

ROOT = Path(__file__).parent
BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://magicpin-ai-challenge-roan.vercel.app").rstrip("/")


def call(method: str, path: str, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(BASE + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=25) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


call("POST", "/v1/teardown", {})
items = [
    ("category", "dentists", load("dataset/categories/dentists.json")),
    ("merchant", "m_001_drmeera_dentist_delhi", load("dataset/merchants/m_001_drmeera_dentist_delhi.json")),
    ("customer", "c_001_priya_for_m001", load("dataset/customers/c_001_priya_for_m001.json")),
    ("trigger", "trg_003_recall_due_priya", load("dataset/triggers/trg_003_recall_due_priya.json")),
]
for scope, context_id, payload in items:
    status, result = call("POST", "/v1/context", {
        "scope": scope, "context_id": context_id, "version": 1,
        "payload": payload, "delivered_at": "2026-04-26T00:00:00Z",
    })
    assert status == 200 and result["accepted"], (scope, result)

_, health = call("GET", "/v1/healthz")
assert health["contexts_loaded"] == {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}, health

tick_body = {"now": "2026-04-26T10:30:00Z", "available_triggers": ["trg_003_recall_due_priya"]}
_, first = call("POST", "/v1/tick", tick_body)
assert len(first["actions"]) == 1 and "₹299" in first["actions"][0]["body"], first
_, second = call("POST", "/v1/tick", tick_body)
assert second["actions"] == [], second

reply_base = {"merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
              "from_role": "merchant", "received_at": "2026-04-26T10:45:00Z"}
auto = "Thank you for contacting us! Our team will respond shortly."
_, wait = call("POST", "/v1/reply", {**reply_base, "conversation_id": "smoke-auto-1", "message": auto, "turn_number": 2})
_, end = call("POST", "/v1/reply", {**reply_base, "conversation_id": "smoke-auto-2", "message": auto, "turn_number": 3})
assert wait["action"] == "wait" and end["action"] == "end", (wait, end)

call("POST", "/v1/teardown", {})
print(json.dumps({"ok": True, "base": BASE, "health": health, "message": first["actions"][0]["body"]}, ensure_ascii=True))
