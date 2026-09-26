# Context First - Vera submission

Vera is a context-grounded FastAPI assistant for the Magicpin AI Challenge. It
supports both submission modes: the live HTTP harness and the generated
`submission.jsonl`. The service stores versioned category, merchant, customer,
and trigger context; selects relevant events; composes category-specific
WhatsApp messages; and handles follow-up conversation turns.

## Why this design

Known challenge cases use deterministic templates with source lookup. This
keeps composition fast and repeatable; incomplete evidence gets a conservative
response. The policy checks expiry, recipient identity, trigger-matched consent, opt-outs
and deduplication before outbound actions. Replies deliver the requested draft
without claiming to publish posts, book appointments or charge a subscription.
An optional Vercel AI Gateway fallback is limited to unfamiliar injected
trigger types and ambiguous replies. Its result must satisfy a strict schema,
numeric grounding checks, and length limits; if any check or network call fails,
Vera uses the deterministic result. The LLM adapter is **disabled in production**:
this Vercel account requires payment-card verification before Gateway will run.

On Vercel, contexts, suppression keys, and conversation state live in Upstash
Redis (Vercel Marketplace, free tier), scoped to the deployment. Each context
write is one Lua script that compares versions and stores atomically across
workers; health counts come from per-scope sets. Teardown deletes only that
deployment's keys. Private Vercel Blob (`blob_state.py`) remains a fallback, but
Hobby Blob allows only 2,000 put/list operations per 30 days, fewer than one
judge run needs. Local development uses memory. The conversation layer
also detects repeated WhatsApp Business auto-replies, ends on opt-outs or
abuse, executes direct commitments without re-qualifying, waits when asked, and
declines unrelated professional advice. Customer outreach requires dated consent
for the trigger's purpose, a matching merchant and a WhatsApp channel. The brief's
schema needs no legacy `reminder_opt_in` flag, but an explicit false flag or
revoked consent blocks outreach. Unknown customer trigger purposes fail closed.
Health counts never substitute hardcoded warmup totals. Conversation and tick
mutations serialize within a worker, but multi-record operations are not
distributed transactions; simultaneous ticks for the same recipient remain a
limitation. Do not redeploy during a judging session (state is deployment-scoped).

## Conversation layer update (26 Sep 2026, deployed before the 20:30 IST cutoff)

A judge-style local harness (`stress_harness.py`; outputs in `artifacts/stress-before.txt`
and `artifacts/stress-after.txt`) showed that most free-form merchant replies fell through to one
generic routing question, and that unseen trigger kinds produced a one-field message. The reply
handler now answers questions from trigger and merchant data, delivers the draft on any commitment
("ok", "theek hai", "chalo"), closes gracefully after delivery, catches verbatim-repeat auto-replies,
de-escalates abuse once, redirects off-topic requests, answers cost questions truthfully, applies
"make it shorter", and mirrors Hinglish per turn. Unseen trigger kinds state their payload facts,
anchor on the merchant's own snapshot and offer one concrete deliverable. 48 tests pass; the
30 canonical `submission.jsonl` messages are unchanged.

The magicpin briefs are not included in this repository; `dataset/` and `judge_simulator.py` are kept
so the tests and simulator can run.

## Run and test

```powershell
python -m pip install -r requirements.txt
python -m uvicorn bot:app --host 0.0.0.0 --port 8080
python build_submission.py
python -m unittest -v test_bot.py test_blob_state.py test_redis_state.py   # 48 tests
```

`submission.jsonl` is generated from `dataset/test_pairs.json`; regenerate it
after changing composition rules. The included `judge_simulator.py` can target
`http://localhost:8080`.

The deployed root path is a human-readable status dashboard. `/docs` exposes
the interactive OpenAPI explorer, while the challenge contract lives under
`/v1/*`.

Live URL: https://magicpin-ai-challenge-roan.vercel.app
First-touch template: `vera_contextual_v1`, body `{{1}}`; the single parameter
is the full composed message. This is a challenge template, not a Meta approval.

## Configuration

- `TEAM_NAME`, `TEAM_MEMBERS`, `CONTACT_EMAIL`, `SUBMITTED_AT`: metadata.
- `KV_REST_API_URL`, `KV_REST_API_TOKEN`: Upstash Redis (preferred), connected via Vercel Marketplace.
- `BLOB_READ_WRITE_TOKEN`: private Vercel Blob store (fallback only).
- `VERA_LLM_ENABLED=1`: allow the adaptive AI fallback.
- `VERA_LLM_MODEL`: Gateway model; defaults to `openai/gpt-5.4-mini`.
- `AI_GATEWAY_API_KEY` or Vercel's automatic `VERCEL_OIDC_TOKEN`: Gateway auth.

The most valuable future inputs would be live booking inventory, a verified
offer source of truth, and explicit approval state for each outreach template.
Vera intentionally does not infer these facts.

Storage implementation references: [consistent private reads](https://vercel.com/changelog/vercel-blob-now-supports-consistent-reads-on-private-storage)
and [conditional writes](https://vercel.com/docs/vercel-blob#conditional-writes).
