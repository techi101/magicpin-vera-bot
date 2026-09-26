import json
import copy
import unittest
from unittest.mock import patch, AsyncMock
from pathlib import Path

from fastapi.testclient import TestClient

import bot


ROOT = Path(__file__).parent
DATA = ROOT / "dataset"


def load_dir(folder: str, key: str) -> dict[str, dict]:
    result = {}
    for path in (DATA / folder).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[payload[key]] = payload
    return result


class VeraContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(bot.app)
        cls.categories = load_dir("categories", "slug")
        cls.merchants = load_dir("merchants", "merchant_id")
        cls.customers = load_dir("customers", "customer_id")
        cls.triggers = load_dir("triggers", "id")

    def setUp(self):
        bot.contexts.clear()
        bot.conversations.clear()
        bot.sent_suppressions.clear()
        bot.auto_reply_fingerprints.clear()
        bot.local_state.clear()

    def push(self, scope: str, context_id: str, payload: dict, version: int = 1):
        return self.client.post("/v1/context", json={
            "scope": scope, "context_id": context_id, "version": version,
            "payload": payload, "delivered_at": "2026-04-26T00:00:00Z",
        })

    def test_warmup_counts_and_idempotency(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("Merchant intelligence", root.text)
        for slug, payload in self.categories.items():
            self.assertTrue(self.push("category", slug, payload).json()["accepted"])
        for mid, payload in self.merchants.items():
            self.assertTrue(self.push("merchant", mid, payload).json()["accepted"])
        for cid, payload in self.customers.items():
            self.assertTrue(self.push("customer", cid, payload).json()["accepted"])
        health = self.client.get("/v1/healthz").json()
        self.assertEqual(health["contexts_loaded"], {
            "category": 5, "merchant": 50, "customer": 200, "trigger": 0,
        })
        duplicate = self.push("category", "dentists", self.categories["dentists"])
        self.assertEqual(duplicate.status_code, 200)
        self.assertTrue(duplicate.json()["accepted"])
        self.assertTrue(duplicate.json()["duplicate"])

    def test_tick_customer_message_and_suppression(self):
        trigger = self.triggers["trg_003_recall_due_priya"]
        merchant = self.merchants[trigger["merchant_id"]]
        customer = self.customers[trigger["customer_id"]]
        self.push("category", merchant["category_slug"], self.categories[merchant["category_slug"]])
        self.push("merchant", merchant["merchant_id"], merchant)
        self.push("customer", customer["customer_id"], customer)
        self.push("trigger", trigger["id"], trigger)
        payload = {"now": "2026-04-26T10:00:00Z", "available_triggers": [trigger["id"]]}
        first = self.client.post("/v1/tick", json=payload).json()["actions"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["send_as"], "merchant_on_behalf")
        self.assertEqual(first[0]["template_name"], "vera_contextual_v1")
        self.assertIn("₹299", first[0]["body"])
        self.assertEqual(self.client.post("/v1/tick", json=payload).json()["actions"], [])

    def test_replay_routes(self):
        base = {"merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
                "from_role": "merchant", "received_at": "2026-04-26T10:00:00Z"}
        auto = "Thank you for contacting us! Our team will respond shortly."
        one = self.client.post("/v1/reply", json={**base, "conversation_id": "a1", "message": auto, "turn_number": 2}).json()
        two = self.client.post("/v1/reply", json={**base, "conversation_id": "a2", "message": auto, "turn_number": 3}).json()
        self.assertEqual(one["action"], "wait")
        self.assertEqual(two["action"], "end")
        go = self.client.post("/v1/reply", json={**base, "conversation_id": "go", "message": "Ok lets do it. Whats next?", "turn_number": 2}).json()
        self.assertEqual(go["action"], "send")
        self.assertIn("Here is the draft", go["body"])
        self.assertNotIn("moved to action", go["body"])
        hostile = self.client.post("/v1/reply", json={**base, "conversation_id": "stop", "message": "Stop messaging me. This is spam.", "turn_number": 2}).json()
        self.assertEqual(hostile["action"], "end")

        first_unknown = self.client.post("/v1/reply", json={**base, "merchant_id": "other", "conversation_id": "unknown", "message": "hmm", "turn_number": 2}).json()
        repeated_unknown = self.client.post("/v1/reply", json={**base, "merchant_id": "other", "conversation_id": "unknown", "message": "okay", "turn_number": 3}).json()
        self.assertEqual(first_unknown["action"], "send")
        self.assertIn("No pressure", first_unknown["body"])
        self.assertEqual(repeated_unknown["action"], "send")  # "okay" after the offer = commitment -> deliver
        closing = self.client.post("/v1/reply", json={**base, "merchant_id": "other", "conversation_id": "unknown", "message": "thanks", "turn_number": 4}).json()
        self.assertEqual(closing["action"], "send")
        after_close = self.client.post("/v1/reply", json={**base, "merchant_id": "other", "conversation_id": "unknown", "message": "ok", "turn_number": 5}).json()
        self.assertEqual(after_close["action"], "end")

    def test_free_form_replies(self):
        base = {"merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
                "from_role": "merchant", "received_at": "2026-04-26T10:00:00Z"}
        def say(conv, msg, turn):
            return self.client.post("/v1/reply", json={**base, "conversation_id": conv, "message": msg, "turn_number": turn}).json()
        canned = "Hello, we are away right now. Leave your number please."
        self.assertNotEqual(say("rep", canned, 2)["action"], "end")
        self.assertIn(say("rep", canned, 3)["action"], {"wait", "end"})
        self.assertEqual(say("rep", canned, 4)["action"], "end")
        abuse = say("ab", "you people are scammers", 2)
        self.assertEqual(abuse["action"], "send")
        self.assertIn("STOP", abuse["body"])
        gst = say("ab", "can you help me file my GST?", 3)
        self.assertEqual(gst["action"], "send")
        self.assertIn("outside what I can help with", gst["body"])
        self.assertEqual(say("ab", "useless bakwas", 4)["action"], "end")
        why = say("q", "why is this important?", 2)
        self.assertEqual(why["action"], "send")
        self.assertIn("YES", why["body"])
        price = say("p", "kitna paisa lagega?", 2)
        self.assertIn("Nothing is charged", price["body"])
        self.assertIn("reply kar dijiye", price["body"])

    def test_unknown_trigger_kind_is_grounded(self):
        merchant = json.loads(Path("dataset/merchants/m_001_drmeera_dentist_delhi.json").read_text(encoding="utf-8"))
        category = json.loads(Path("dataset/categories/dentists.json").read_text(encoding="utf-8"))
        trigger = {"id": "new1", "kind": "competitor_price_drop", "scope": "merchant", "merchant_id": merchant["merchant_id"],
                   "payload": {"service": "dental cleaning", "competitor_price": 249, "distance_km": 1.2}}
        msg = bot.compose(category, merchant, trigger)
        self.assertIn("₹249", msg["body"])
        self.assertIn("1.2 km", msg["body"])
        self.assertEqual(msg["cta"], "binary")
        self.assertNotIn("I noticed", msg["body"])

    def test_invalid_scope_is_400(self):
        response = self.push("unsupported", "x", {})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["reason"], "invalid_scope")

    def test_all_generated_contexts_compose_without_fabrication_sentinels(self):
        for trigger in self.triggers.values():
            merchant = self.merchants[trigger["merchant_id"]]
            category = self.categories[merchant["category_slug"]]
            customer = self.customers.get(trigger.get("customer_id"))
            result = bot.compose(category, merchant, trigger, customer)
            self.assertTrue(result["body"].strip(), trigger["id"])
            self.assertNotIn("None", result["body"], trigger["id"])
            self.assertIn(result["send_as"], {"vera", "merchant_on_behalf"})
            self.assertIn(result["cta"], {"binary", "open_ended", "none"})

    def test_static_submission_contract(self):
        rows = [json.loads(line) for line in (ROOT / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 30)
        self.assertEqual([row["test_id"] for row in rows], [f"T{i:02d}" for i in range(1, 31)])
        for row in rows:
            self.assertTrue({"body", "cta", "send_as", "suppression_key", "rationale"} <= row.keys())

    def seed_recall(self):
        trigger = copy.deepcopy(self.triggers["trg_003_recall_due_priya"])
        merchant = copy.deepcopy(self.merchants[trigger["merchant_id"]])
        customer = copy.deepcopy(self.customers[trigger["customer_id"]])
        category = copy.deepcopy(self.categories[merchant["category_slug"]])
        for scope, cid, payload in [("category", category["slug"], category), ("merchant", merchant["merchant_id"], merchant), ("customer", customer["customer_id"], customer), ("trigger", trigger["id"], trigger)]:
            self.push(scope, cid, payload)
        return category, merchant, trigger, customer

    def tick_for(self, tids, now="2026-04-26T10:00:00Z"):
        response = self.client.post("/v1/tick", json={"now": now, "available_triggers": tids})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["actions"]

    def test_stale_versions_return_409_and_latest_context_wins(self):
        category = copy.deepcopy(self.categories["dentists"])
        self.push("category", "dentists", category, 2)
        stale = self.push("category", "dentists", {**category, "digest": []}, 1)
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(bot.contexts[("category", "dentists")]["payload"], category)

    def test_brief_consent_schema_without_legacy_optin_is_accepted(self):
        _, _, trigger, customer = self.seed_recall()
        customer["preferences"].pop("reminder_opt_in")
        self.push("customer", customer["customer_id"], customer, 2)
        self.assertEqual(len(self.tick_for([trigger["id"]])), 1)

    def test_wrong_consent_scope_and_merchant_are_rejected(self):
        _, _, trigger, customer = self.seed_recall()
        customer["consent"]["scope"] = []
        self.push("customer", customer["customer_id"], customer, 2)
        self.assertEqual(self.tick_for([trigger["id"]]), [])
        customer["consent"]["scope"] = ["promotional_offers"]
        self.push("customer", customer["customer_id"], customer, 3)
        self.assertEqual(self.tick_for([trigger["id"]]), [])
        customer["consent"]["scope"] = ["recall_reminders"]
        customer["preferences"]["reminder_opt_in"] = True
        customer["merchant_id"] = "different_business"
        self.push("customer", customer["customer_id"], customer, 4)
        self.assertEqual(self.tick_for([trigger["id"]]), [])

    def test_health_counts_partial_load_and_versions_exactly(self):
        self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"],
                         {"category": 0, "merchant": 0, "customer": 0, "trigger": 0})
        self.seed_recall()
        expected = {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}
        self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"], expected)
        self.push("category", "dentists", self.categories["dentists"])
        self.push("category", "dentists", self.categories["dentists"], 2)
        self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"], expected)
        self.push("category", "unseen-category", {"slug": "unseen-category"})
        self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"],
                         {**expected, "category": 2})

    def test_remote_health_never_inflates_or_uses_worker_memory(self):
        class PartialCache:
            async def get(self, key):
                return ["one", "one"] if key in {"index:category", "index:customer"} else None
        self.seed_recall()
        with patch.object(bot, "runtime_cache", PartialCache()):
            self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"],
                             {"category": 1, "merchant": 0, "customer": 1, "trigger": 0})

    def test_scoped_consent_is_required_for_every_customer_kind(self):
        customer = {"merchant_id": "m", "preferences": {},
                    "consent": {"opted_in_at": "2025-11-04", "scope": []}}
        for kind, scopes in bot.CUSTOMER_CONSENT_SCOPES.items():
            with self.subTest(kind=kind):
                trigger = {"merchant_id": "m", "kind": kind}
                for scope in scopes:
                    customer["consent"]["scope"] = [scope]
                    self.assertTrue(bot.customer_allowed(customer, trigger))
                customer["consent"]["scope"] = ["unrelated_scope"]
                self.assertFalse(bot.customer_allowed(customer, trigger))
        customer["consent"]["scope"] = ["recall_reminders"]
        self.assertFalse(bot.customer_allowed(customer, {"merchant_id": "m", "kind": "unseen_kind"}))

    def test_consent_optouts_channels_and_malformed_data_fail_closed(self):
        _, _, trigger, original = self.seed_recall()
        for field, value in [("reminder_opt_in", False), ("reminder_opt_in", "true"),
                             ("channel", "sms"), ("channel", "whatsapp_fake")]:
            customer = copy.deepcopy(original)
            customer["preferences"][field] = value
            self.assertFalse(bot.customer_allowed(customer, trigger), (field, value))
        for field, value in [("revoked_at", "2026-01-01"), ("opted_out_at", "2026-01-01"),
                             ("scope", "recall_reminders"), ("scope", [{}]),
                             ("opted_in_at", "invalid"), ("opted_in_at", None)]:
            customer = copy.deepcopy(original)
            customer["consent"][field] = value
            self.assertFalse(bot.customer_allowed(customer, trigger), (field, value))
        for channel in ["whatsapp", "whatsapp_via_parent", "whatsapp_via_son"]:
            customer = copy.deepcopy(original)
            customer["preferences"]["channel"] = channel
            self.assertTrue(bot.customer_allowed(customer, trigger))

    def test_expired_trigger_is_not_sent(self):
        _, _, trigger, _ = self.seed_recall()
        self.assertEqual(self.tick_for([trigger["id"]], "2026-12-01T00:00:00Z"), [])

    def test_same_tick_dedup_and_recipient_scoping(self):
        _, merchant, trigger, _ = self.seed_recall()
        shared = {**trigger, "scope": "merchant", "kind": "curious_ask_due", "customer_id": None, "suppression_key": "category:event"}
        self.push("trigger", shared["id"], shared, 2)
        other_trigger = {**shared, "id": "second", "merchant_id": "second_merchant"}
        other_merchant = {**merchant, "merchant_id": "second_merchant"}
        self.push("merchant", "second_merchant", other_merchant)
        self.push("trigger", "second", other_trigger)
        duplicate = {**shared, "id": "duplicate"}
        self.push("trigger", "duplicate", duplicate)
        actions = self.tick_for([shared["id"], shared["id"], "duplicate", "second"])
        self.assertEqual(len(actions), 2)
        self.assertEqual(len({a["merchant_id"] for a in actions}), 2)
        self.assertEqual(self.tick_for([shared["id"], "second"]), [])
        for a in actions:
            self.assertEqual(a["template_params"], [a["body"]])

    def test_missing_digest_id_does_not_substitute_unrelated_study(self):
        category, merchant, trigger, _ = self.seed_recall()
        trigger.update(scope="merchant", kind="research_digest", customer_id=None, payload={"top_item_id": "unseen"})
        result = bot.compose(category, merchant, trigger)
        self.assertNotIn("38%", result["body"])
        self.assertIn("missing", result["body"])

    def test_inline_and_updated_digest_are_used(self):
        category, merchant, trigger, _ = self.seed_recall()
        item = {"id": "new-item", "kind": "research", "title": "New appointment scheduling study", "summary": "Follow-up attendance rose 17% in the supplied trial.", "source": "Synthetic test bulletin"}
        trigger.update(scope="merchant", kind="category_research_digest_release", customer_id=None, payload={"top_item": item})
        result = bot.compose(category, merchant, trigger)
        self.assertIn("17%", result["body"])
        self.assertIn("Synthetic test bulletin", result["body"])
        category["digest"].append(item)
        self.push("category", category["slug"], category, 2)
        trigger["payload"] = {"top_item_id": "new-item"}
        self.push("trigger", trigger["id"], trigger, 2)
        self.assertIn("17%", self.tick_for([trigger["id"]])[0]["body"])

    def test_reply_delivers_research_summary_and_is_idempotent(self):
        category, merchant, trigger, _ = self.seed_recall()
        trigger.update(scope="merchant", customer_id=None, kind="research_digest", payload={"top_item_id": category["digest"][0]["id"]})
        self.push("trigger", trigger["id"], trigger, 2)
        action = self.tick_for([trigger["id"]])[0]
        reply = {"conversation_id": action["conversation_id"], "merchant_id": merchant["merchant_id"], "from_role": "merchant", "message": "Yes, send me the summary", "received_at": "2026-04-26T10:01:00Z", "turn_number": 2}
        one = self.client.post("/v1/reply", json=reply).json()
        two = self.client.post("/v1/reply", json=reply).json()
        self.assertEqual(one, two)
        self.assertIn("38%", one["body"])
        self.assertIn("Customer-share draft", one["body"])
        self.assertNotIn("moved to action", one["body"])

    def test_negation_and_deferral_beat_positive_keywords(self):
        for i, msg in enumerate(["Don't proceed yet", "Yes, but later", "Not ready to join"]):
            reply = {"conversation_id": f"negative-{i}", "merchant_id": "x", "from_role": "merchant", "message": msg, "received_at": "2026-04-26T10:01:00Z", "turn_number": 2}
            result = self.client.post("/v1/reply", json=reply).json()
            self.assertEqual(result["action"], "wait", result)

    def test_optout_suppresses_future_ticks(self):
        _, merchant, trigger, _ = self.seed_recall()
        trigger.update(scope="merchant", customer_id=None, kind="curious_ask_due")
        self.push("trigger", trigger["id"], trigger, 2)
        reply = {"conversation_id": "stop-test", "merchant_id": merchant["merchant_id"], "from_role": "merchant", "message": "Stop messaging me", "received_at": "2026-04-26T10:01:00Z", "turn_number": 2}
        self.assertEqual(self.client.post("/v1/reply", json=reply).json()["action"], "end")
        self.assertEqual(self.tick_for([trigger["id"]]), [])

    def test_appointment_uses_confirmation_and_actual_timestamp(self):
        category, merchant, trigger, customer = self.seed_recall()
        trigger.update(kind="appointment_tomorrow", payload={"appointment_time": "2026-11-05T18:00:00+05:30", "service": "cleaning"})
        result = bot.compose(category, merchant, trigger, customer)
        self.assertNotIn("recall", result["body"])
        self.assertIn("6:00pm", result["body"])

    def test_seed_slot_weekday_derived_from_iso(self):
        category, merchant, trigger, customer = self.seed_recall()
        result = bot.compose(category, merchant, trigger, customer)
        self.assertIn("Thu 5 Nov 2026", result["body"])
        self.assertNotIn("Wed 5 Nov", result["body"])

    def test_sparse_performance_and_zero_renewal_are_safe(self):
        category, merchant, trigger, _ = self.seed_recall()
        trigger.update(scope="merchant", kind="perf_dip", payload={})
        self.assertNotIn("None", bot.compose(category, merchant, trigger)["body"])
        trigger.update(kind="renewal_due", payload={"days_remaining": 0})
        self.assertIn("0 days", bot.compose(category, merchant, trigger)["body"])

    def test_validation_errors_remain_json_400(self):
        response = self.client.post("/v1/tick", json={"now": "nonsense"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["reason"], "malformed_request")

    def test_teardown_clears_context_and_optout(self):
        self.seed_recall()
        bot.local_state["optout:x:merchant"] = True
        self.assertTrue(self.client.post("/v1/teardown").json()["wiped"])
        self.assertEqual(sum(self.client.get("/v1/healthz").json()["contexts_loaded"].values()), 0)
        self.assertEqual(bot.local_state, {})

    def test_cache_miss_does_not_resurrect_worker_memory(self):
        class EmptyCache:
            async def get(self, key): return None
        self.seed_recall()
        with patch.object(bot, "runtime_cache", EmptyCache()):
            self.assertEqual(sum(self.client.get("/v1/healthz").json()["contexts_loaded"].values()), 0)
            self.assertEqual(self.tick_for(["trg_003_recall_due_priya"]), [])

    def test_cache_failure_does_not_acknowledge_context_write(self):
        class BrokenCache:
            async def get(self, key): raise RuntimeError("simulated outage")
        with patch.object(bot, "runtime_cache", BrokenCache()):
            self.assertEqual(self.push("category", "dentists", self.categories["dentists"]).status_code, 503)

    def test_blob_health_reports_actual_counts_and_fails_closed(self):
        storage = AsyncMock()
        storage.counts.return_value = {"category": 1, "merchant": 2, "customer": 3, "trigger": 4}
        with patch.object(bot, "blob_state", storage):
            self.assertEqual(self.client.get("/v1/healthz").json()["contexts_loaded"], storage.counts.return_value)
            storage.counts.side_effect = RuntimeError("outage")
            self.assertEqual(self.client.get("/v1/healthz").status_code, 503)

    def test_blob_versions_use_atomic_backend_result(self):
        storage = AsyncMock()
        stored = {"version": 2, "payload": self.categories["dentists"], "stored_at": "2026-04-26T10:00:00Z"}
        storage.put_context.side_effect = [("accepted", stored), ("duplicate", stored), ("stale", stored)]
        with patch.object(bot, "blob_state", storage):
            self.assertTrue(self.push("category", "dentists", stored["payload"], 2).json()["accepted"])
            self.assertTrue(self.push("category", "dentists", stored["payload"], 2).json()["duplicate"])
            stale = self.push("category", "dentists", stored["payload"], 1)
            self.assertEqual(stale.status_code, 409)
            self.assertEqual(stale.json()["current_version"], 2)
        self.assertEqual(bot.local_state, {})

    def test_blob_failure_does_not_acknowledge_write_or_wipe(self):
        storage = AsyncMock()
        storage.put_context.side_effect = RuntimeError("write outage")
        storage.clear.side_effect = RuntimeError("wipe outage")
        with patch.object(bot, "blob_state", storage):
            self.assertEqual(self.push("category", "dentists", self.categories["dentists"]).status_code, 503)
            self.assertEqual(self.client.post("/v1/teardown").status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
