"""Production-only probe for the unfamiliar-trigger adaptive path."""
import json
from pathlib import Path
from urllib import request

ROOT = Path(__file__).parent
BASE = "https://magicpin-ai-challenge-roan.vercel.app"


def call(path: str, payload: dict):
    req = request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


call("/v1/teardown", {})
try:
    inputs = [
        ("category", "dentists", load("dataset/categories/dentists.json")),
        ("merchant", "m_001_drmeera_dentist_delhi",
         load("dataset/merchants/m_001_drmeera_dentist_delhi.json")),
        ("trigger", "trg_adaptive_probe", {
            "id": "trg_adaptive_probe",
            "scope": "merchant",
            "kind": "local_demand_shift",
            "source": "external",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "customer_id": None,
            "payload": {
                "search_topic": "emergency dentist near me",
                "locality": "Lajpat Nagar",
                "signal": "rising",
            },
            "urgency": 2,
            "suppression_key": "probe:adaptive:v3",
            "expires_at": "2026-10-01T00:00:00Z",
        }),
    ]
    for scope, context_id, payload in inputs:
        result = call("/v1/context", {
            "scope": scope,
            "context_id": context_id,
            "version": 1,
            "payload": payload,
            "delivered_at": "2026-09-25T00:00:00Z",
        })
        assert result["accepted"], result

    result = call("/v1/tick", {
        "now": "2026-09-25T12:00:00Z",
        "available_triggers": ["trg_adaptive_probe"],
    })
    assert len(result["actions"]) == 1, result
    action = result["actions"][0]
    used_model = action["rationale"] != (
        "New trigger type handled conservatively using only its supplied payload."
    )
    print(json.dumps({
        "ok": True,
        "adaptive_model_used": used_model,
        "body": action["body"],
        "rationale": action["rationale"],
    }, ensure_ascii=False))
finally:
    call("/v1/teardown", {})
