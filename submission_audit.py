"""Reproducible HTTP acceptance audit. Resets the target challenge bot's test state.

Usage: python submission_audit.py --base URL --output artifacts/live-audit.json
No LLM score is generated; these are executable acceptance checks only.
"""
import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).parent


def run(base):
    timings = []
    checks = []
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("bot.py", "blob_state.py", "redis_state.py")}

    def call(method, path, payload=None, expected=200, as_json=True):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(base + path, method=method, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
        started = time.monotonic()
        try:
            response = request.urlopen(req, timeout=29)
        except error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read().decode("utf-8")
            allowed = (expected,) if isinstance(expected, int) else expected
            assert response.status in allowed, (path, response.status, raw)
        timings.append(time.monotonic() - started)
        return json.loads(raw) if as_json else raw

    def check(name, condition):
        assert condition, name
        checks.append(name)
        print("PASS " + name, flush=True)

    def push(scope, cid, payload, version=1, expected=200):
        return call("POST", "/v1/context", {"scope": scope, "context_id": cid, "version": version, "payload": payload, "delivered_at": "2026-04-26T10:00:00Z"}, expected)

    def tick(ids, now="2026-04-26T10:00:00Z"):
        return call("POST", "/v1/tick", {"now": now, "available_triggers": ids})["actions"]

    def reply(conv, message, turn, mid, customer_id=None):
        return call("POST", "/v1/reply", {"conversation_id": conv, "merchant_id": mid, "customer_id": customer_id, "from_role": "customer" if customer_id else "merchant", "message": message, "turn_number": turn, "received_at": "2026-04-26T10:01:00Z"})

    call("POST", "/v1/teardown", {})
    try:
        check("Homepage is HTML", "Merchant intelligence" in call("GET", "/", as_json=False))
        metadata = call("GET", "/v1/metadata")
        check("Metadata identity populated", bool(metadata["contact_email"] and metadata["team_members"]))
        check("Deployed source matches the submission files", metadata.get("source_sha256") == source_hashes)
        if ".vercel.app" in base:
            check("Production uses Redis state (Blob's Hobby op quota cannot cover a judge run)", metadata.get("state_backend") == "redis")
        contexts = {}
        records = []
        for scope, folder, field in [("category", "categories", "slug"), ("merchant", "merchants", "merchant_id"), ("customer", "customers", "customer_id")]:
            contexts[scope] = {}
            for path in sorted((ROOT / "dataset" / folder).glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                contexts[scope][payload[field]] = payload
                records.append((scope, payload[field], payload))
        for scope in contexts:
            cid, payload = next(iter(contexts[scope].items()))
            push(scope, cid, payload)
        check("Partial warmup reports actual counts, not expected totals",
              call("GET", "/v1/healthz")["contexts_loaded"] ==
              {"category": 1, "merchant": 1, "customer": 1, "trigger": 0})
        # Five concurrent connections; batches are paced below the 10 RPS contract.
        with ThreadPoolExecutor(max_workers=5) as pool:
            for offset in range(0, len(records), 5):
                results = list(pool.map(lambda item: push(*item), records[offset:offset + 5]))
                assert all(x["accepted"] for x in results)
                if (offset + 5) % 50 == 0:
                    print(f"Warmup: {offset + 5}/{len(records)}", flush=True)
                time.sleep(0.55)
        expected_counts = {"category": 5, "merchant": 50, "customer": 200, "trigger": 0}
        check("Full 255-context concurrent warmup", call("GET", "/v1/healthz")["contexts_loaded"] == expected_counts)
        # A duplicate acknowledgement proves the record survived; a fresh write
        # would silently repair a missing record, so require duplicate on EVERY ID.
        with ThreadPoolExecutor(max_workers=5) as pool:
            for offset in range(0, len(records), 5):
                results = list(pool.map(lambda item: push(*item), records[offset:offset + 5]))
                assert all(x.get("duplicate") is True for x in results), "Acknowledged context lost before readback"
                if (offset + 5) % 50 == 0:
                    print(f"Readback: {offset + 5}/{len(records)}", flush=True)
                time.sleep(0.55)
        check("All 255 context versions survive readback", True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda v: push("category", "audit-concurrent", {"slug": "audit-concurrent", "revision": v}, v, (200, 409)), [2, 5, 3, 4, 1]))
        check("Concurrent versions preserve highest version",
              push("category", "audit-concurrent", {"slug": "audit-concurrent"}, 5)["duplicate"] and
              push("category", "audit-concurrent", {"slug": "audit-concurrent"}, 4, 409)["current_version"] == 5)
        check("Concurrent versions count as one context",
              call("GET", "/v1/healthz")["contexts_loaded"] == {**expected_counts, "category": 6})

        category = contexts["category"]["dentists"]
        check("Same-version push is idempotent", push("category", "dentists", category)["duplicate"])
        new_category = copy.deepcopy(category)
        item = {"id": "audit-new-study", "kind": "research", "title": "Follow-up scheduling study", "summary": "The supplied test study reports 17% more follow-up attendance.", "source": "Synthetic audit bulletin, April 2026"}
        new_category["digest"].append(item)
        push("category", "dentists", new_category, 2)
        check("Older version returns HTTP 409", not push("category", "dentists", category, 1, 409)["accepted"])
        mid = "m_001_drmeera_dentist_delhi"
        trigger = {"id": "audit-research", "scope": "merchant", "kind": "research_digest", "merchant_id": mid, "payload": {"top_item_id": item["id"]}, "urgency": 2, "suppression_key": "audit-study", "expires_at": "2026-05-01T00:00:00Z"}
        push("trigger", trigger["id"], trigger)
        actions = tick([trigger["id"], trigger["id"]])
        check("Injected research and same-tick dedup", len(actions) == 1 and "17%" in actions[0]["body"])
        check("Cross-request dedup", tick([trigger["id"]]) == [])
        continuation = reply(actions[0]["conversation_id"], "Yes, send the summary", 2, mid)
        check("Reply returns requested source and draft", continuation["action"] == "send" and "17%" in continuation["body"] and "Customer-share draft" in continuation["body"])
        check("Reply retry is idempotent", reply(actions[0]["conversation_id"], "Yes, send the summary", 2, mid) == continuation)

        recall = json.loads((ROOT / "dataset/triggers/trg_003_recall_due_priya.json").read_text(encoding="utf-8"))
        customer = copy.deepcopy(contexts["customer"][recall["customer_id"]])
        # Test the brief's schema, which does NOT include this legacy flag.
        customer["preferences"].pop("reminder_opt_in", None)
        customer["consent"]["scope"] = ["promotional_offers"]
        push("customer", customer["customer_id"], customer, 2)
        push("trigger", recall["id"], recall)
        check("Unrelated consent cannot authorize recall", tick([recall["id"]]) == [])
        customer["consent"]["scope"] = ["recall_reminders"]
        customer["preferences"]["reminder_opt_in"] = False
        push("customer", customer["customer_id"], customer, 3)
        check("Explicit opt-out overrides matching scope", tick([recall["id"]]) == [])
        customer["preferences"].pop("reminder_opt_in")
        push("customer", customer["customer_id"], customer, 4)
        recall_actions = tick([recall["id"]])
        check("Consent-scope schema permits legitimate recall", len(recall_actions) == 1 and recall_actions[0]["send_as"] == "merchant_on_behalf")
        check("ISO slot weekdays are correct", "Thu 5 Nov 2026" in recall_actions[0]["body"])
        expired = {**trigger, "id": "audit-expired", "suppression_key": "audit-expired", "expires_at": "2026-04-25T00:00:00Z"}
        push("trigger", expired["id"], expired)
        check("Expired event produces no send", tick([expired["id"]]) == [])
        check("Negated intent does not execute", reply("audit-negative", "Don't proceed yet", 2, mid)["action"] == "wait")
        auto = "Thank you for contacting us! Our team will respond shortly."
        check("First auto-reply waits", reply("audit-auto1", auto, 2, mid)["action"] == "wait")
        check("Repeated auto-reply ends", reply("audit-auto2", auto, 3, mid)["action"] == "end")
        check("Opt-out ends", reply("audit-stop", "Stop messaging me", 2, mid)["action"] == "end")
        future = {**trigger, "id": "audit-after-stop", "suppression_key": "audit-after-stop"}
        push("trigger", future["id"], future)
        check("Opt-out suppresses future trigger", tick([future["id"]]) == [])
    finally:
        call("POST", "/v1/teardown", {})
    check("Teardown leaves zero contexts", sum(call("GET", "/v1/healthz")["contexts_loaded"].values()) == 0)
    return {"passed": True, "base_url": base, "tested_at": datetime.now(timezone.utc).isoformat(), "version": metadata["version"], "checks": checks, "requests": len(timings), "max_request_seconds": round(max(timings), 3), "judge_score": None, "bot_sha256": source_hashes["bot.py"], "source_sha256": source_hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", default="artifacts/live-audit.json")
    args = parser.parse_args()
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never leave an old green report behind when a newer run fails.
    report = {"passed": False, "base_url": args.base.rstrip("/"),
              "tested_at": datetime.now(timezone.utc).isoformat(),
              "bot_sha256": hashlib.sha256((ROOT / "bot.py").read_bytes()).hexdigest(),
              "error": "Audit incomplete", "judge_score": None}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        report = run(args.base.rstrip("/"))
    except Exception as exc:
        report["error"] = str(exc)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
