# Magicpin submission handoff

Public endpoint: https://magicpin-ai-challenge-roan.vercel.app

Team: Context First  
Member: Nitesh Kumar Yadav, NSUT, roll no. 2023UEA7232  
Contact (college email): nitesh.kumar.ug23@nsut.ac.in  
Version: 2.2.1
Runtime: Python 3.12 / FastAPI on Vercel

## What to submit

Submit the **base URL above** in the organizer's portal. If file uploads are
requested, use `magicpin-submission-v2.2.1.zip` (its live audit passed). For
individual files, include `bot.py`, its `redis_state.py` and `blob_state.py` modules, `requirements.txt`,
`submission.jsonl` and `README.md`. The ZIP also contains dependency
pins, tests, deployment configuration and the supplied synthetic dataset for
reproduction. `MANIFEST.json` inside it records each included file's SHA-256.

## Suggested approach description

Vera consumes versioned category, merchant, trigger and customer context. Its
deterministic composer retrieves the referenced source and selects a concise,
category-specific message. A policy layer checks trigger expiry, matching
customer consent, opt-outs and recipient-scoped duplicate suppression. Replies
deliver drafts and source summaries immediately after acceptance; booking,
publication and subscription actions remain explicitly pending external
confirmation. The hosted service persists test state in Upstash Redis, uses an
atomic version check for every context write, and clears its records on
teardown. The optional LLM adapter is disabled in this release.

## Verification and limits

- The old `magicpin-submission-v2.2.0.zip` is superseded: its health counts
  overstated partial loads and its consent rules did not match the testing brief.
  Do not submit that archive.
- A daily Vercel cron calls `/v1/healthz` (which reads Redis) so the free
  Upstash database never goes idle long enough to be archived.
- Storage moved from Vercel Blob to Upstash Redis on 2026-09-26: Hobby Blob's
  2,000 put/list operations per 30 days were 90% used by our own audits and
  could not cover a judge run.
- Final hosted audit passed on 2026-09-26 (04:15 UTC, Redis backend): 26
  checks, 551 requests, slowest request 0.83 s, including the full 255-context concurrent
  warmup, readback of every record, and updates to large records. The supplied
  simulator's four non-LLM scenarios (warmup, auto-reply, intent, hostile) also
  pass against the live URL.
- Self-measured quality (not an official score): the supplied judge_simulator
  scorer, run with Groq `openai/gpt-oss-120b` on the 30 canonical pairs, gave
  36.8/50 on average (baseline before the 2026-09-26 composer pass: 34.0/50).
  That pass rewrote sparse-trigger messages (IPL, recall, refill, lapsed
  customers, festival, perf dip/spike, competitor, milestone) to lead with
  verified merchant numbers and a yes/no ask, and fixed broken Hinglish CTAs.
  Run-to-run judge noise is about ±5 per message. Details are in
  `artifacts/self-judge-scores.json`.
- Storage writes retry transient upstream failures (5xx, 429, timeouts), which
  were observed on about 2% of Blob writes. Retries are idempotent: a write that
  already landed resolves as a duplicate.

- `test_bot.py`, `test_blob_state.py` and `test_redis_state.py` cover contract behavior, composition,
  consent, cross-worker version races, storage failures and cleanup. Run with
  `python -m unittest -v test_bot.py test_blob_state.py test_redis_state.py`.
- `artifacts/live-audit.json` records the hosted acceptance checks and latency.
- `run_simulator_contract.py` runs the original simulator's non-LLM scenarios.
  It does not generate an LLM evaluation score.
- No selection probability or official judge score is claimed.
- Context versions use cross-worker conditional writes; health counts list
  actual context objects. Conversation/tick operations still lack distributed
  multi-record transactions. Do not redeploy during the test session.
- Gateway inference remains disabled because this account requires card
  verification. No billing change was made.
- The seed dataset contains inconsistent display weekdays. The composer uses
  supplied ISO timestamps for slot weekdays without changing the original data.

## Submission status

The project is deployed; **portal submission is not yet confirmed**. The supplied
briefs mention a portal but include no URL, closing date, or completed logistics.
No form has been submitted, recruiter contacted, or receipt received. The live
`/v1/metadata` endpoint reports the member and college email above.
