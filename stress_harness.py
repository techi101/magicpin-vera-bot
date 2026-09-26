"""Local judge-style stress test: unseen trigger kinds, injected context, free-form replies.

Runs the FastAPI app in-process with in-memory state (no Redis env), prints every bot output.
Usage: python stress_harness.py > artifacts/stress-output.txt
"""
import glob
import json
import os
import sys

for key in ("KV_REST_API_URL", "KV_REST_API_TOKEN", "BLOB_READ_WRITE_TOKEN", "VERA_LLM_ENABLED"):
    os.environ.pop(key, None)
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient  # noqa: E402

import bot  # noqa: E402

client = TestClient(bot.app)
NOW = "2026-04-26T10:00:00Z"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def push(scope, cid, payload, version=1):
    r = client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": version,
                                         "payload": payload, "delivered_at": NOW})
    assert r.status_code == 200, (scope, cid, r.status_code, r.text)


categories = {load(f)["slug"]: load(f) for f in glob.glob("dataset/categories/*.json")}
merchants = {load(f)["merchant_id"]: load(f) for f in glob.glob("dataset/merchants/*.json")}
customers = {load(f)["customer_id"]: load(f) for f in glob.glob("dataset/customers/*.json")}
client.post("/v1/teardown")
for slug, c in categories.items(): push("category", slug, c)
for mid, m in merchants.items(): push("merchant", mid, m)
for cid, c in customers.items(): push("customer", cid, c)
print("healthz", client.get("/v1/healthz").json()["contexts_loaded"])


def fire(trigger, label):
    push("trigger", trigger["id"], trigger)
    r = client.post("/v1/tick", json={"now": NOW, "available_triggers": [trigger["id"]]}).json()
    acts = r.get("actions", [])
    print(f"\n=== {label} [{trigger['kind']}] ===")
    if not acts:
        print("  (no action)")
        return None
    a = acts[0]
    print("  BOT:", a["body"])
    print("  cta:", a.get("cta"), "| why:", a.get("rationale"))
    return a


def converse(action, replies, label, role="merchant"):
    if not action:
        return
    print(f"--- replies: {label}")
    for i, text in enumerate(replies, start=2):
        r = client.post("/v1/reply", json={
            "conversation_id": action["conversation_id"], "merchant_id": action["merchant_id"],
            "customer_id": action.get("customer_id"), "from_role": role, "message": text,
            "received_at": NOW, "turn_number": i}).json()
        print(f"  {role.upper()}: {text}")
        print(f"  BOT[{r.get('action')}]: {r.get('body') or r.get('rationale')}")
        if r.get("action") == "end":
            break


M_DEN, M_REST, M_SAL, M_GYM, M_PH = ("m_001_drmeera_dentist_delhi", "m_006_southindiancafe_restaurant_bangalore",
                                     "m_003_studio11_salon_hyderabad", "m_008_zenyoga_gym_chennai",
                                     "m_010_sunrisepharm_pharmacy_pune")
M_PH = next((m for m in merchants if "pharm" in m), M_PH)

# 1. Unseen trigger kinds (Phase 3 injects 15 new triggers)
unseen = [
    {"id": "x1", "scope": "merchant", "kind": "footfall_drop", "merchant_id": M_REST,
     "payload": {"metric": "store_visits", "delta_pct": -0.22, "window": "14d", "vs_baseline": 140}, "urgency": 3},
    {"id": "x2", "scope": "merchant", "kind": "new_review_posted", "merchant_id": M_SAL,
     "payload": {"rating": 2, "review_text": "Waited 40 minutes past my booking time", "posted_at": "2026-04-25T18:00:00Z"}, "urgency": 4},
    {"id": "x3", "scope": "merchant", "kind": "search_trend_spike", "merchant_id": M_GYM,
     "payload": {"query": "morning yoga near me", "locality": "Adyar", "delta_pct": 0.41, "window": "7d"}, "urgency": 2},
    {"id": "x4", "scope": "merchant", "kind": "stock_out_risk", "merchant_id": M_PH,
     "payload": {"molecule": "metformin 500mg", "days_of_cover": 3, "weekly_demand_units": 120}, "urgency": 4},
    {"id": "x5", "scope": "merchant", "kind": "competitor_price_drop", "merchant_id": M_DEN,
     "payload": {"service": "dental cleaning", "competitor_price": 249, "distance_km": 1.2}, "urgency": 3},
]
acts = {t["id"]: fire(t, "UNSEEN") for t in unseen}

# 2. Injected digest item (new category version) + research trigger that references it
cat = json.loads(json.dumps(categories["dentists"]))
cat["digest"].append({"id": "d_new_aligner", "kind": "research", "title": "Clear aligners show 18% faster finishing in adults vs brackets",
                      "source": "AJO-DO Mar 2026", "trial_n": 640, "summary": "Adults 25-40 finished 18% faster with aligners; relapse similar at 12 months.",
                      "actionable": "Mention aligner timelines during adult consults"})
push("category", "dentists", cat, version=2)
a_dig = fire({"id": "x6", "scope": "merchant", "kind": "research_digest", "merchant_id": "m_002_bharat_dentist_mumbai",
              "payload": {"category": "dentists", "top_item_id": "d_new_aligner"}, "urgency": 2}, "INJECTED DIGEST")

# 3. Performance snapshot update then perf_dip
m = json.loads(json.dumps(merchants[M_SAL]))
m["performance"].update({"views": 1111, "calls": 7, "ctr": 0.0063, "delta_7d": {"views_pct": -0.31, "calls_pct": -0.42}})
push("merchant", M_SAL, m, version=2)
a_perf = fire({"id": "x7", "scope": "merchant", "kind": "perf_dip", "merchant_id": M_SAL,
               "payload": {"metric": "calls", "delta_pct": -0.42, "window": "7d", "vs_baseline": 12}, "urgency": 4}, "UPDATED PERF")

# 4. New customer + recall_due 2 min later
new_c = {"customer_id": "c_new_1", "merchant_id": M_DEN,
         "identity": {"name": "Arjun", "language_pref": "hi-en mix", "age_band": "30-40"},
         "relationship": {"first_visit": "2025-10-01", "last_visit": "2025-10-20", "visits_total": 2, "services_received": ["scaling"]},
         "state": "active", "preferences": {"channel": "whatsapp"},
         "consent": {"opted_in_at": "2025-10-01", "scope": ["recall_reminders"]}}
push("customer", "c_new_1", new_c)
a_rec = fire({"id": "x8", "scope": "customer", "kind": "recall_due", "merchant_id": M_DEN, "customer_id": "c_new_1",
              "payload": {"service_due": "6_month_scaling", "due_date": "2026-04-28",
                          "available_slots": [{"iso": "2026-04-28T11:00:00+05:30", "label": "Tue 28 Apr, 11am"},
                                              {"iso": "2026-04-29T17:00:00+05:30", "label": "Wed 29 Apr, 5pm"}]},
              "urgency": 3}, "NEW CUSTOMER RECALL")

# 5. Free-form merchant replies of the kind a sub-LLM produces
base = [load(f) for f in sorted(glob.glob("dataset/triggers/*.json"))]
def seed(idx):
    t = dict(base[idx]); t["id"] = f"{t['id']}_r{idx}"; t["expires_at"] = None; t["suppression_key"] = t["id"]
    return fire(t, f"SEED {idx}")

converse(seed(0), ["Interesting. Tell me more?", "How would I use this with patients?", "ok sounds good"], "engaged-curious")
converse(seed(3), ["hmm why did calls drop?", "what should I do then", "theek hai, karo"], "engaged-hinglish")
converse(seed(9), ["Thank you for your message. We are currently closed and will reply soon."] * 4, "auto-reply-no-marker")
converse(seed(11), ["you people are scammers, always asking money", "can you also help me file my GST?", "ok whatever"], "hostile-then-offtopic")
converse(seed(12), ["which offices? how many people?", "okay", "sure go ahead"], "qualify-then-commit")
converse(seed(23), ["Kitna paisa lagega?", "accha, aur kya karna hoga", "chalo done"], "hindi-price")
converse(acts.get("x2"), ["yes please draft a reply", "can you make it shorter?", "perfect thanks"], "unseen-review")
converse(acts.get("x1"), ["what is store visits?", "I am not sure", "maybe next week"], "unseen-hesitant")
converse(a_rec, ["2", "thanks"], "customer-slot", role="customer")
