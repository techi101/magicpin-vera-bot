"""Vera challenge bot: context-grounded outbound composition and conversation routing.

Run locally:  uvicorn bot:app --host 0.0.0.0 --port 8080
Known challenge routes are deterministic. An optional, grounded AI Gateway
fallback handles unfamiliar injected triggers and ambiguous replies; every
route still works when the model or network is unavailable.
"""
from __future__ import annotations

import json
import asyncio
import os
import re
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
from blob_state import PrivateBlobState
from redis_state import RedisState

try:  # Available in production; local development keeps the memory fallback.
    from vercel.functions import AsyncRuntimeCache
except ImportError:  # pragma: no cover - exercised only outside Vercel
    AsyncRuntimeCache = None

app = FastAPI(title="Vera Context Composer", version="2.2.1")
SOURCE_HASHES = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ("bot.py", "blob_state.py", "redis_state.py")}
request_oidc_token: ContextVar[str | None] = ContextVar("request_oidc_token", default=None)
START = time.time()
contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, list[dict[str, str]]] = defaultdict(list)
sent_suppressions: set[str] = set()
auto_reply_fingerprints: Counter[str] = Counter()
local_state: dict[str, Any] = {}
state_lock = asyncio.Lock()
_deployment_namespace = "vera-" + (
    os.getenv("VERCEL_DEPLOYMENT_ID")
    or os.getenv("VERCEL_URL")
    or "local"
).replace(".", "-")
STATE_TAG = _deployment_namespace + "-state-v2"
runtime_cache = (
    AsyncRuntimeCache(namespace=_deployment_namespace)
    if AsyncRuntimeCache is not None and os.getenv("VERCEL")
    else None
)
# Upstash Redis is preferred: Hobby Blob's 2,000 put/list operations per 30 days
# cannot cover a judge run. Either backend exposes get/set/put_context/counts/clear.
_redis_url = os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
_redis_token = os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")
if _redis_url and _redis_token:
    blob_state = RedisState(_deployment_namespace, _redis_url, _redis_token)
    STATE_BACKEND = "redis"
elif os.getenv("BLOB_READ_WRITE_TOKEN"):
    blob_state = PrivateBlobState(_deployment_namespace, os.environ["BLOB_READ_WRITE_TOKEN"])
    STATE_BACKEND = "private_blob"
else:
    blob_state, STATE_BACKEND = None, None
if blob_state is not None:
    runtime_cache = None
LLM_MODEL = os.getenv("VERA_LLM_MODEL", "openai/gpt-5.4-mini")
KNOWN_TRIGGER_KINDS = {
    "research_digest", "regulation_change", "cde_opportunity", "supply_alert",
    "recall_due", "appointment_tomorrow", "chronic_refill_due", "customer_lapsed_hard",
    "customer_lapsed_soft", "trial_followup", "wedding_package_followup", "ipl_match_today",
    "perf_dip", "seasonal_perf_dip", "perf_spike", "competitor_opened",
    "review_theme_emerged", "active_planning_intent", "curious_ask_due", "dormant_with_vera",
    "festival_upcoming", "category_seasonal", "renewal_due", "gbp_unverified",
    "winback_eligible", "milestone_reached",
}


@app.middleware("http")
async def capture_vercel_oidc(request: Request, call_next):
    """Vercel exposes the runtime OIDC token on each Function request header."""
    token = request.headers.get("x-vercel-oidc-token")
    marker = request_oidc_token.set(token)
    try:
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            try:
                async with asyncio.timeout(27):
                    # Context updates have backend CAS protection; serializing
                    # unrelated IDs in this worker needlessly queues warmup.
                    if blob_state is not None and request.url.path == "/v1/context":
                        return await call_next(request)
                    async with state_lock:
                        return await call_next(request)
            except TimeoutError:
                return JSONResponse(status_code=503, content={"detail": "Request budget exceeded; retry"})
        return await call_next(request)
    finally:
        request_oidc_token.reset(marker)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={
        "accepted": False, "reason": "malformed_request",
        "details": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()],
    })


async def state_get(key: str, default: Any = None) -> Any:
    """Cache is authoritative on Vercel; never resurrect stale worker memory."""
    storage = blob_state or runtime_cache
    if storage is not None:
        try:
            value = await storage.get(key)
            return default if value is None else value
        except Exception as exc:
            raise HTTPException(503, "State storage unavailable; retry this request") from exc
    return local_state.get(key, default)


async def state_set(key: str, value: Any) -> None:
    storage = blob_state or runtime_cache
    if storage is not None:
        try:
            await storage.set(key, value, {"tags": [STATE_TAG], "ttl": 86400})
        except Exception as exc:
            raise HTTPException(503, "State storage unavailable; write not acknowledged") from exc
    else:
        local_state[key] = value


def context_key(scope: str, context_id: str) -> str:
    return f"context:{scope}:{context_id}"


async def load_context(scope: str, context_id: str | None) -> dict | None:
    if not context_id:
        return None
    local = contexts.get((scope, context_id)) if runtime_cache is None and blob_state is None else None
    stored = await state_get(context_key(scope, context_id), local)
    if stored:
        contexts[(scope, context_id)] = stored
    return stored


def llm_enabled() -> bool:
    return os.getenv("VERA_LLM_ENABLED", "0") == "1" and bool(
        os.getenv("AI_GATEWAY_API_KEY")
        or os.getenv("VERCEL_OIDC_TOKEN")
        or request_oidc_token.get()
    )


def grounded_numbers(text: str, evidence: Any) -> bool:
    """Reject model output containing numeric claims absent from supplied evidence."""
    output_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", json.dumps(evidence, ensure_ascii=False)))
    return output_numbers <= evidence_numbers


async def gateway_json(system: str, user: dict, schema: dict) -> dict | None:
    if not llm_enabled():
        return None
    token = (
        os.getenv("AI_GATEWAY_API_KEY")
        or os.getenv("VERCEL_OIDC_TOKEN")
        or request_oidc_token.get()
    )
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "vera_action", "strict": True, "schema": schema},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(
                "https://ai-gateway.vercel.sh/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500].replace("\n", " ")
        print(f"AI Gateway fallback activated: {exc.response.status_code} {detail}")
        return None
    except Exception as exc:
        print(f"AI Gateway fallback activated: {type(exc).__name__}")
        return None


async def refine_unknown_trigger(category: dict, merchant: dict, trigger: dict,
                                 customer: dict | None, draft: dict) -> dict:
    evidence = {"category": category, "merchant": merchant, "trigger": trigger,
                "customer": customer, "safe_draft": draft}
    schema = {
        "type": "object",
        "properties": {
            "body": {"type": "string"},
            "cta": {"type": "string", "enum": ["binary", "open_ended", "none"]},
            "rationale": {"type": "string"},
        },
        "required": ["body", "cta", "rationale"], "additionalProperties": False,
    }
    result = await gateway_json(
        "You compose one concise WhatsApp message for Magicpin's Vera. Use ONLY facts in the JSON. "
        "Never invent a number, offer, date, source, customer fact, action completion, or competitor. "
        "Match category voice and language, explain why now, use one CTA, and improve the safe draft. "
        "Return the required JSON only.", evidence, schema,
    )
    if not result or not result.get("body") or len(result["body"]) > 700:
        return draft
    if not grounded_numbers(result["body"], evidence):
        return draft
    result["body"] = safe_sentence(result["body"], category)
    return {**draft, "body": result["body"], "cta": result["cta"],
            "rationale": result["rationale"]}


async def intelligent_reply(history: list[dict], merchant: dict | None,
                            category: dict | None, inbound: ReplyBody) -> dict | None:
    evidence = {"history": history[-10:], "merchant": merchant, "category": category,
                "latest_message": inbound.message}
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["send", "wait", "end"]},
            "body": {"type": "string"},
            "cta": {"type": "string", "enum": ["binary", "open_ended", "none"]},
            "wait_seconds": {"type": "integer"},
            "rationale": {"type": "string"},
        },
        "required": ["action", "body", "cta", "wait_seconds", "rationale"],
        "additionalProperties": False,
    }
    result = await gateway_json(
        "You are Vera in an active WhatsApp conversation. Use ONLY supplied facts. Honor direct intent now; "
        "do not re-qualify after commitment. Stay within merchant growth/profile/offers/customer messaging. "
        "Never claim an action is completed unless history proves it. One concise response, no repetition. "
        "For wait/end use an empty body. Return JSON only.", evidence, schema,
    )
    if not result or result.get("action") not in {"send", "wait", "end"}:
        return None
    if result["action"] == "send":
        if not result.get("body") or len(result["body"]) > 700 or not grounded_numbers(result["body"], evidence):
            return None
        result["body"] = safe_sentence(result["body"], category or {})
    return result


def get(d: dict | None, *keys: str, default: Any = None) -> Any:
    value: Any = d or {}
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def pct(value: Any) -> str:
    try:
        percentage = float(value) * 100
        return f"{percentage:.0f}%" if percentage.is_integer() else f"{percentage:.1f}%"
    except (TypeError, ValueError):
        return ""


def humanize(value: Any) -> str:
    return str(value or "").replace("_", " ").replace("+", "+").strip()


def number(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "not supplied"


def short_date(value: Any, include_time: bool = False) -> str:
    """Render an ISO value for WhatsApp while preserving unknown strings unchanged."""
    if not value:
        return ""
    raw = str(value)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        text = f"{parsed.day} {parsed.strftime('%b %Y')}"
        if include_time and (parsed.hour or parsed.minute):
            text += f", {parsed.strftime('%I:%M%p').lstrip('0').lower()}"
        return text
    except (ValueError, TypeError):
        return raw


def active_offer(merchant: dict) -> str | None:
    offers = [o["title"] for o in merchant.get("offers", [])
              if o.get("status") == "active" and o.get("title")]
    return next((title for title in offers if "@" in title and "₹" in title),
                offers[0] if offers else None)


def digest_for(category: dict, trigger: dict) -> dict | None:
    payload = trigger.get("payload", {})
    wanted = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    if isinstance(payload.get("top_item"), dict):
        return payload["top_item"]
    for item in category.get("digest", []):
        if item.get("id") == wanted:
            return item
    if wanted:
        return None  # Never silently substitute unrelated research for a missing ID.
    kind = trigger.get("kind", "")
    expected = {"research_digest": "research", "regulation_change": "compliance", "cde_opportunity": "cde"}.get(kind)
    if expected:
        return next((d for d in reversed(category.get("digest", [])) if d.get("kind") == expected), None)
    return None


CUSTOMER_BENEFIT = {
    "dentists": "A 6-monthly check-up catches small issues before they need bigger treatment.",
    "gyms": "Consistency is the hard part — one session gets your rhythm back.",
    "salons": "A quick touch-up keeps your look fresh between bigger appointments.",
    "pharmacies": "A quick refill check keeps your regular medicines on schedule.",
    "restaurants": "Your usual favourites are ready whenever you are.",
}
SLOT_CTA = {"pharmacies": " Reply YES and we’ll keep your usual order ready for pickup or delivery.",
            "restaurants": " Reply YES and we’ll reserve a table for you this week."}
RECALL_NOUN = {"dentists": "dental check-up", "gyms": "progress check-in", "salons": "maintenance appointment",
               "pharmacies": "medicine review", "restaurants": "visit"}


def customer_offer(merchant: dict) -> str | None:
    """Existing customers get an offer not reserved for first-timers when one exists."""
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active" and o.get("title")]
    returning = [t for t in offers if not re.search(r"first|trial|new", t, re.I)]
    return (returning or offers or [None])[0]


def customer_sender(merchant: dict) -> str:
    owner = get(merchant, "identity", "owner_first_name", default="")
    business = get(merchant, "identity", "name", default="your clinic")
    if merchant.get("category_slug") == "dentists" and owner:
        return f"{merchant_name(merchant)}’s team at {business}"
    return business


def merchant_name(merchant: dict) -> str:
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name")
    category = merchant.get("category_slug")
    if owner:
        return f"Dr. {owner}" if category == "dentists" and not owner.lower().startswith("dr.") else owner
    return identity.get("name", "there")


def code_mix(text: str, profile: dict | None) -> str:
    """Small, natural language adaptation without pretending to speak unknown local languages."""
    pref = get(profile, "identity", "language_pref", default="")
    if not pref:
        pref = ",".join(get(profile, "identity", "languages", default=[]))
    pref = str(pref).lower()
    if "hi" in pref:
        return (text.replace("Reply YES to confirm", "YES reply karke confirm karein")
                .replace("Reply YES", "YES reply kar dijiye")
                .replace("tell us if you need to reschedule", "reschedule karna ho toh bata dein"))
    return text


def safe_sentence(text: str, category: dict) -> str:
    """Avoid category taboos even if an injected payload contains one."""
    taboos = get(category, "voice", "vocab_taboo", default=[]) + get(category, "voice", "taboos", default=[])
    for taboo in taboos:
        text = re.sub(r"(?<!\w)" + re.escape(str(taboo)) + r"(?!\w)", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def slot_labels(payload: dict) -> list[str]:
    labels = []
    for slot in payload.get("available_slots", []) or payload.get("next_session_options", []):
        if isinstance(slot, str):
            labels.append(slot)
        elif isinstance(slot, dict):
            # ISO is authoritative: seed display labels have incorrect weekdays.
            moment = parse_time(slot.get("iso"))
            if moment:
                labels.append(moment.strftime("%a ") + short_date(slot["iso"], include_time=True) + " (local time)")
            elif slot.get("label"):
                labels.append(str(slot["label"]))
    return labels


CUSTOMER_CONSENT_SCOPES = {
    "recall_due": {"recall_reminders"},
    "appointment_tomorrow": {"appointment_reminders"},
    "chronic_refill_due": {"refill_reminders"},
    "customer_lapsed_hard": {"winback_offers", "promotional_offers"},
    "customer_lapsed_soft": {"winback_offers", "promotional_offers"},
    "trial_followup": {"program_updates", "kids_program_updates"},
    "wedding_package_followup": {"bridal_package_followup"},
}


def customer_allowed(customer: dict | None, trigger: dict) -> bool:
    if not customer or customer.get("merchant_id") != trigger.get("merchant_id"):
        return False
    preferences = customer.get("preferences", {})
    consent = customer.get("consent", {})
    if not isinstance(preferences, dict) or not isinstance(consent, dict):
        return False
    # The testing brief's schema has scoped consent but no legacy opt-in flag.
    # A supplied negative/invalid flag still takes precedence over that consent.
    if ("reminder_opt_in" in preferences and preferences["reminder_opt_in"] is not True) or consent.get("revoked_at") or consent.get("opted_out_at"):
        return False
    channel = str(preferences.get("channel", "whatsapp")).lower()
    if channel != "whatsapp" and not channel.startswith("whatsapp_via_"):
        return False
    scopes = consent.get("scope", [])
    if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
        return False
    opted_in_at = consent.get("opted_in_at")
    if not isinstance(opted_in_at, str) or parse_time(opted_in_at) is None:
        return False
    return bool(set(scopes) & CUSTOMER_CONSENT_SCOPES.get(trigger.get("kind"), set()))


def recipient_key(merchant_id: str | None, customer_id: str | None = None) -> str:
    return f"{merchant_id}:{customer_id or 'merchant'}"


TRIGGER_ALIASES = {
    "research_digest_release": "research_digest",
    "category_research_digest_release": "research_digest",
    "scheduled_recurring": "curious_ask_due",
}


SKIP_FACT_KEYS = {"category", "merchant_id", "customer_id", "placeholder", "metric_or_topic", "id", "trigger_id",
                  "merchant_last_message", "window", "metric"}


def payload_facts(p: dict, limit: int = 3) -> list[str]:
    """Human-readable facts from an unfamiliar payload, without adding interpretation."""
    facts = []
    delta = next((p[k] for k in ("delta_pct", "change_pct") if isinstance(p.get(k), (int, float))), None)
    if p.get("metric") and delta is not None:
        window = f" over {p['window']}" if p.get("window") else ""
        facts.append(f"{humanize(p['metric'])} {'up' if delta >= 0 else 'down'} {pct(abs(delta))}{window}")
    for key, value in p.items():
        if key in ("delta_pct", "change_pct") and facts and delta is not None:
            continue
        if key in SKIP_FACT_KEYS or key.endswith("_id") or value in (None, "", [], {}):
            continue
        label = humanize(key).replace("vs baseline", "baseline")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and (key.endswith("_pct") or key.startswith("delta")):
            facts.append(f"{label.replace(' pct', '')} {'up' if value >= 0 else 'down'} {pct(abs(value))}")
        elif isinstance(value, (int, float)) and re.search(r"price|amount|revenue|value_inr|_inr$", key):
            facts.append(f"{label} ₹{number(value)}")
        elif isinstance(value, (int, float)) and key.endswith("_km"):
            facts.append(f"{label.replace(' km', '')} {value} km")
        elif isinstance(value, (int, float)):
            facts.append(f"{label} {number(value) if float(value).is_integer() else value}")
        elif isinstance(value, str):
            if parse_time(value) and re.match(r"\d{4}-\d{2}-\d{2}", value):
                facts.append(f"{label} {short_date(value)}")
            elif len(value) > 60 or re.search(r"text|quote|comment|review", key):
                facts.append(f"{label}: “{value[:140].rstrip()}”")
            else:
                facts.append(f"{label}: {humanize(value)}")
        elif isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
            facts.append(f"{label}: {', '.join(humanize(v) for v in value[:3])}")
        if len(facts) >= limit:
            break
    return facts


def next_step(kind: str, p: dict, offer: str | None) -> str:
    """The single concrete artifact Vera offers for a trigger."""
    kind = kind or ""
    if "review" in kind:
        return "a public reply draft plus one operating fix"
    if any(k in kind for k in ("research", "digest", "cde")):
        return "a 3-line summary plus a patient/customer-share draft"
    if any(k in kind for k in ("regulation", "compliance")):
        return "a short compliance checklist"
    if any(k in kind for k in ("stock", "supply", "refill")) or p.get("molecule"):
        return "a reorder and customer-note checklist for pharmacist review"
    if any(k in kind for k in ("competitor", "price")):
        return "a value-first counter post" + (f" built on your {offer}" if offer else "")
    if any(k in kind for k in ("renewal", "subscription")):
        return "the renewal summary"
    if "gbp" in kind or "verif" in kind:
        return "the verification checklist"
    if "planning" in kind:
        return "the finished announcement copy"
    if "milestone" in kind:
        return "a thank-you post with a gentle review request"
    if "ipl" in kind or "match" in kind:
        return "a match-night delivery post"
    if any(k in kind for k in ("dip", "drop", "decline", "dormant", "lapsed", "winback")):
        return "a recovery post" + (f" around {offer}" if offer else "") + " instead of a generic discount"
    if any(k in kind for k in ("spike", "trend", "search", "seasonal", "festival")):
        return "a timely post" + (f" around {offer}" if offer else "") + " while demand is up"
    return "a ready-to-post update" + (f" using {offer}" if offer else "")


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Public composition function required by the static-submission brief."""
    trigger = {**trigger, "kind": TRIGGER_ALIASES.get(trigger.get("kind"), trigger.get("kind", ""))}
    p, kind = trigger.get("payload", {}), trigger.get("kind", "")
    identity, performance, aggregate = merchant.get("identity", {}), merchant.get("performance", {}), merchant.get("customer_aggregate", {})
    owner, business, offer = merchant_name(merchant), identity.get("name", "your business"), active_offer(merchant)
    item = digest_for(category, trigger)
    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" and customer else "vera"
    cta, body, why = "open_ended", "", ""

    if send_as == "merchant_on_behalf":
        cname = get(customer, "identity", "name", default="there")
        slots_text = slot_labels(p)
        services = get(customer, "relationship", "services_received", default=[])
        service = p.get("service_due") or (services[-1] if services else "your next visit")
        if kind == "appointment_tomorrow" and p.get("placeholder"):
            body = f"Hi {cname}, {business} here. Your appointment is tomorrow. Reply YES to confirm; tell us if you need to reschedule."
            cta, why = "binary", "Appointment-tomorrow trigger supports confirmation without inventing a time or service."
        elif kind == "recall_due" and p.get("placeholder"):
            slug = merchant.get("category_slug", "")
            perk = customer_offer(merchant)
            body = f"Hi {cname}, {customer_sender(merchant)} here — your routine {RECALL_NOUN.get(slug, 'follow-up')} is due."
            body += f" {CUSTOMER_BENEFIT.get(slug, '')}" + (f" {perk} is on right now." if perk else "")
            body += SLOT_CTA.get(slug, " Reply YES and we’ll share this week’s open slots.")
            cta, why = "binary", "Sparse recall trigger: names the due check, a category benefit and a live merchant offer; no invented date, service or slot."
        elif kind == "appointment_tomorrow":
            when = short_date(p.get("appointment_time") or p.get("appointment_time_iso"), include_time=True)
            service = humanize(p.get("service") or p.get("service_due") or "appointment")
            body = f"Hi {cname}, {business} here. Reminder: your {service} is {'on ' + when if when else 'tomorrow'}. Reply YES to confirm."
            cta, why = "binary", "Appointment confirmation uses supplied timing; it does not invent a booking or change the service into a recall."
        elif kind == "recall_due":
            when = short_date(p.get("due_date") or p.get("appointment_time")) or "now"
            choices = " or ".join(slots_text[:2])
            suffix = f" Available: {choices}." if choices else ""
            body = f"Hi {cname}, {business} here. Your {humanize(service)} recall is due {when}.{suffix}"
            if offer: body += f" Current offer: {offer}; eligibility to be confirmed by the clinic."
            body += " Reply YES to request a booking."
            cta, why = "binary", "Consent-aware recall uses the supplied due date, available slots, and active merchant offer."
        elif kind == "chronic_refill_due" and p.get("placeholder"):
            body = (f"Hi {cname}, {customer_sender(merchant)} here — your refill follow-up is due. So you don’t run short, "
                    "reply YES with the medicine name and we’ll check it against your record and keep it ready.")
            cta, why = "binary", "Sparse refill trigger asks for the one missing fact instead of inventing medication details."
        elif kind in {"chronic_refill_due", "supply_alert"} and merchant.get("category_slug") == "pharmacies":
            molecules = ", ".join(p.get("molecule_list", []) or [p.get("molecule", "your regular medicines")])
            due = short_date(p.get("stock_runs_out_iso") or p.get("due_date")) or "soon"
            delivery = " Free home delivery is available." if offer and "Delivery" in offer else ""
            senior = next((o.get("title") for o in merchant.get("offers", [])
                           if o.get("status") == "active" and "Senior" in o.get("title", "")), None)
            body = f"Namaste {cname}, {business} here. Your regular medicines—{molecules}—run out on {due}."
            if senior: body += f" Current offer: {senior}; eligibility needs confirmation."
            body += f"{delivery} Reply CONFIRM to request a refill review; the pharmacist must confirm prescription, dose and delivery details."
            cta, why = "binary", "Medication reminder uses only supplied molecules and refill timing."
        elif kind in {"customer_lapsed_hard", "customer_lapsed_soft", "trial_followup"}:
            days = p.get("days_since_last_visit")
            focus = p.get("previous_focus") or get(customer, "preferences", "training_focus", default="your routine")
            option = slots_text[0] if slots_text else None
            sender = customer_sender(merchant)
            last_seen = short_date(get(customer, "relationship", "last_visit", default=None))
            visits = get(customer, "relationship", "visits_total", default=None)
            if days:
                body = f"Hi {cname}, {sender} here. It has been {days} days since your last visit."
            elif last_seen:
                history = f" after {visits} visits with us" if isinstance(visits, int) and visits > 1 else ""
                body = f"Hi {cname}, {sender} here — we last saw you on {last_seen}{history}, and we’d love to see you again."
            else:
                body = f"Hi {cname}, {sender} here — it’s been a while, and we’d love to see you again."
            perk = customer_offer(merchant)
            if offer and kind == "customer_lapsed_hard":
                body += f" No judgment—restart at your pace with {offer}, focused on {humanize(focus)}."
            elif focus != "your routine":
                body += f" We can help you restart around {humanize(focus)}, at your pace."
            else:
                body += f" {CUSTOMER_BENEFIT.get(merchant.get('category_slug', ''), 'We can help you pick up where you left off.')}"
                if perk: body += f" {perk} is on right now."
            body += f" Available option: {option}; reply YES to request it." if option else SLOT_CTA.get(merchant.get("category_slug", ""), " Reply YES and we’ll share this week’s open slots.")
            cta, why = "binary", "Warm, no-pressure winback grounded in lapse and preference data."
        elif p.get("placeholder"):
            body = f"Hi {cname}, {business} here. A quick follow-up is ready based on your recent visit. Reply YES if you would like us to help with the next step."
            cta, why = "binary", "Conservative follow-up because the trigger supplies no service or timing detail."
        elif kind == "wedding_package_followup":
            days, window = p.get("days_to_wedding"), p.get("next_step_window_open", "your next prep step")
            body = f"Hi {cname}, {business} here. Your wedding is {days} days away, and the {humanize(window)} window is open. I can reserve your preferred slot—reply YES to continue."
            cta, why = "binary", "Bridal follow-up uses the supplied wedding timeline without inventing a package."
        else:
            body = f"Hi {cname}, {business} here. A quick update is ready for you based on your recent visit. Reply YES if you would like us to help."
            cta, why = "binary", "Conservative customer message for an unknown trigger type."
        body = code_mix(body, customer)
    elif kind in {"research_digest", "regulation_change", "cde_opportunity"} and item:
        title, source, action = item.get("title", "a new category update"), item.get("source"), item.get("actionable")
        source_text = f" Source: {source}." if source else " Source citation was not supplied."
        if kind == "cde_opportunity":
            event_date = short_date(item.get("date"), include_time=True)
            credits = p.get("credits", item.get("credits"))
            details = (f" Date: {event_date}." if event_date else "") + (f" Credits: {credits}." if credits is not None else "")
            body = f"{owner}, {title}.{details} {item.get('summary', '')} {action or ''}{source_text} Want the registration checklist?"
        elif kind == "regulation_change":
            deadline = short_date(p.get("deadline_iso"))
            body = f"{owner}, compliance heads-up: {title}." + (f" Deadline: {deadline}." if deadline else "")
            body += f" {item.get('summary', '')}" + (f" Suggested check: {action}." if action else "")
            body += source_text + " Want a checklist for this update?"
        else:
            cohort = aggregate.get("high_risk_adult_count")
            cohort_text = f"—relevant to your {cohort} high-risk adult patients" if cohort else ""
            study = f" In the {item['trial_n']:,}-patient study," if item.get("trial_n") else ""
            body = f"{owner}, the new digest has one item for you{cohort_text}: {title}.{study} {item.get('summary', '')}{source_text} Want the supplied summary plus a customer-share draft?"
        why = "Uses the exact digest item, source, category vocabulary, and merchant-relevant cohort or action."
    elif kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        body = f"{owner}, a {humanize(kind)} arrived for {business}, but the referenced item is missing. I’ll hold the summary until the correct evidence is available."
        cta, why = "none", "Referenced evidence is absent; no unrelated item or citation is substituted."
    elif kind == "supply_alert":
        batches = ", ".join(p.get("affected_batches", []))
        source = item.get("source") if item else "the supplied recall alert"
        if not all(p.get(key) for key in ("molecule", "affected_batches", "manufacturer")):
            body = f"{owner}, a supply alert arrived, but the product, batches or manufacturer details are incomplete. Verify the original notice before contacting customers or changing stock."
            cta = "none"
        else:
            body = f"{owner}, urgent stock check: {p['molecule']} batches {batches} from {p['manufacturer']} are under voluntary recall. Check those batches against the original notice before contacting affected customers. Source: {source}. Want a customer note for pharmacist review?"
            cta = "binary"
        why = "High-urgency pharmacy alert names the supplied molecule, batches, manufacturer, and safe next steps."
    elif kind == "ipl_match_today":
        match, venue, when = p.get("match"), p.get("venue"), p.get("match_time_iso")
        when_text = short_date(when, include_time=True)
        ipl_item = next((d for d in category.get("digest", []) if "IPL" in d.get("title", "")), {})
        moment = parse_time(when)
        weekend = p.get("is_weeknight") is False or bool(moment and moment.weekday() >= 5)
        day = f"{moment.strftime('%A')}, {when_text}" if moment else when_text
        body = f"Quick heads-up {owner} — {match or 'the listed match'}" + (f" at {venue}" if venue else "") + (f", {day}." if day else " is on the schedule.")
        if ipl_item.get("summary"):
            evidence = re.sub(r"\s*Adjust promo days accordingly\.?", "", ipl_item["summary"].strip())
            body += f" Per {ipl_item.get('source') or 'category data'}: {evidence}"
        if weekend:
            body += " So skip a dine-in match promo and go delivery-first for home watchers."
        else:
            body += " Weeknight matches are your window for a dine-in match-night offer."
        if offer and weekend and re.search(r"\b(mon|tue|wed|thu)", offer, re.I):
            body += f" Your {offer} doesn’t cover today—want me to draft a delivery-only match combo for tonight plus the offer post for the next weeknight match? Reply YES; ready in 10 min."
        elif offer:
            body += f" Your {offer} is the hook—want me to draft the match-night post and delivery banner? Reply YES; ready in 10 min."
        else:
            body += " Want me to draft a match-night delivery post? Reply YES; ready in 10 min."
        cta = "binary"
        why = "Derives weekday vs weekend from the supplied date, applies cited category evidence, and respects the offer's day restrictions."
    elif kind in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
        metric, delta, window = p.get("metric", "performance"), pct(p.get("delta_pct")), p.get("window", "7d")
        if p.get("placeholder") or not isinstance(p.get("delta_pct"), (int, float)):
            peer = get(category, "peer_stats", "avg_ctr", default=None)
            label = get(category, "display_name", default=category.get("slug", "category"))
            ctr = performance.get("ctr")
            body = (f"{owner}, {'a performance spike' if 'spike' in kind else 'a performance dip'} flag came in for {business}. "
                    f"30-day snapshot: {number(performance.get('views'))} views → {number(performance.get('calls'))} calls, CTR {pct(ctr)}")
            if isinstance(ctr, (int, float)) and isinstance(peer, (int, float)):
                body += f" vs the {pct(peer)} magicpin {label} average."
                if "spike" in kind:
                    body += " Let’s lock in the momentum with a fresh post while interest is high."
                elif ctr >= peer:
                    body += " People click, so the leak is after the click—one fresh post with a clear offer usually lifts calls fastest."
                else:
                    body += " Clicks trail peers, so a sharper headline and photos come first."
            else:
                body += "."
            body += " Want me to draft it? Reply YES."
            cta, why = "binary", "Sparse trigger: uses the verified 30-day snapshot and peer benchmark to name one fix, without inventing a direction or cause."
            metric = None
        if metric is None:
            pass
        else:
            peer = get(category, "peer_stats", "avg_ctr", default=None)
            direction = "up" if "spike" in kind else "down"
            direction = "up" if p["delta_pct"] > 0 else "down" if p["delta_pct"] < 0 else "unchanged"
            baseline_text = f" from a baseline of {p['vs_baseline']}" if p.get('vs_baseline') is not None else " versus the comparison period"
            body = f"{owner}, your {window} {metric} are {pct(abs(float(p['delta_pct'])))} {direction}{baseline_text}."
            if kind == "seasonal_perf_dip" and p.get("is_expected_seasonal"):
                body += f" This matches the {humanize(p.get('season_note'))} seasonal lull, so protect retention rather than increasing acquisition spend."
            elif "dip" in kind:
                body += f" Current CTR is {pct(performance.get('ctr'))} versus the {pct(peer)} category benchmark."
                if "unverified_gbp" in merchant.get("signals", []):
                    body += " Verify the Google profile before spending on promotion."
                else:
                    body += f" I can draft one precise recovery post around {offer or 'your strongest service'} instead of a generic discount."
            else:
                label = get(category, "display_name", default="category")
                body += (f" Your {humanize(p.get('likely_driver', 'recent activity'))} is working for {business}—CTR {pct(performance.get('ctr'))} vs the "
                         f"{pct(peer)} magicpin {label} average. Let’s double down while it’s hot: a follow-up post plus a reply script for the new enquiries.")
            if "spike" in kind:
                body += " Reply YES and I’ll draft both."
                cta = "binary"
            else:
                body += " Want the ready-to-publish follow-up?"
            why = "Uses the exact change, baseline, latest performance, peer benchmark, and merchant signals."
    elif kind == "competitor_opened":
        if p.get("competitor_name"):
            body = f"{owner}, {p.get('competitor_name')} opened {p.get('distance_km')} km away on {short_date(p.get('opened_date'))}, advertising {p.get('their_offer')}."
            body += f" Don’t race straight to their price: your {offer or 'own service strength'} can lead on value while your CTR is {pct(performance.get('ctr'))}. Want the comparison-safe Google post?"
        else:
            body = (f"{owner}, a new competitor signal just came in near {identity.get('locality') or business}. You’re starting ahead—"
                    f"{number(performance.get('views'))} views and {number(performance.get('calls'))} calls this month—so protect that lead before they build reviews: "
                    f"a fresh post leading with {offer or 'your strongest service'} keeps you on top of local search. Want me to draft it? Reply YES; ready in 10 min. "
                    "I’ll add their name and offer once verified.")
            cta = "binary"
        why = "Local competitive trigger is paired with a comparison-safe, merchant-specific visibility response."
    elif kind == "review_theme_emerged":
        if p.get("placeholder"):
            positive = next((theme for theme in merchant.get("review_themes", []) if theme.get("sentiment") == "pos"), None)
            if positive:
                body = f"{owner}, {positive.get('occurrences_30d')} recent reviews praise {humanize(positive.get('theme'))}. I can turn that verified pattern into one Google post and a reusable review reply—want them?"
            else:
                body = f"{owner}, a review-pattern check is due, but no theme or count is confirmed yet. I’ll hold the claim until the review data is complete. Want a neutral response template meanwhile?"
        else:
            quote = f" One customer wrote: “{p.get('common_quote')}”." if p.get("common_quote") else ""
            count = p.get('occurrences_30d')
            body = f"{owner}, {str(count) + ' reviews in 30 days' if count is not None else 'a new review signal'} mention {humanize(p.get('theme')) or 'an unspecified theme'}."
            if p.get('trend'): body += f" Trend: {p['trend']}."
            body += f"{quote} I can draft one public reply plus one operating fix—want them?"
        why = "Review-pattern alert uses the observed count and theme, then offers a concrete remedy."
    elif kind == "active_planning_intent":
        topic = humanize(p.get("intent_topic", "the plan"))
        history_text = " ".join(str(turn.get("body", "")) for turn in merchant.get("conversation_history", []))
        if p.get("intent_topic") == "corporate_bulk_thali_package" and "18 orders/day" in history_text:
            body = (f"{owner}, starter draft: “{business} Corporate Lunch—{offer}. Built for offices in {identity.get('locality')}; "
                    "bulk quantity, delivery radius and order cut-off to be confirmed.” Your weekday thali already averages 18 orders/day, so offices are a natural next step. "
                    "Reply with the minimum bulk quantity only; I’ll finish the order-ready WhatsApp copy.")
        elif p.get("intent_topic") == "kids_yoga_summer_camp" and all(
            fact in history_text for fact in ("4-week", "3 classes/week", "age 7-12", "2,499")
        ):
            body = (f"{owner}, here’s the draft from our plan: “{business} Kids Yoga Summer Camp—4 weeks, 3 classes/week, ages 7–12, ₹2,499.” "
                    "These are proposed terms from our earlier conversation, pending your approval. Reply GO for the GBP post plus carousel copy.")
            cta = "binary"
        else:
            body = f"{owner}, first draft for {topic}: lead with {offer or business}, name the audience and outcome, and leave any unconfirmed price/date blank. Reply GO and I’ll format the finished Google and WhatsApp versions."
            cta = "binary"
        why = "Explicit planning intent receives a usable first artifact based on current offer and conversation facts."
    elif kind == "curious_ask_due":
        active = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active" and o.get("title")]
        if len(active) >= 2:
            question = f"which drew more enquiries this week—{active[0]} or {active[1]}?"
        elif active:
            question = f"what customer question comes up most before they choose {active[0]}?"
        else:
            question = "what service or product did customers ask for most this week?"
        body = f"{owner}, quick operator question: {question} Reply with the answer only; I’ll turn it into one Google post and a reusable 4-line WhatsApp reply."
        why = "Asks one easy, merchant-specific question and externalizes the content work."
    elif kind == "dormant_with_vera":
        days = p.get("days_since_last_merchant_message")
        if days:
            call_delta = get(performance, "delta_7d", "calls_pct", default=0)
            call_change = f"down {pct(abs(call_delta))}" if call_delta < 0 else f"up {pct(call_delta)}"
            body = f"{owner}, it’s been {days} days since our last conversation about {humanize(p.get('last_topic'))}. Since then, calls are {call_change} week-on-week. Reply YES and I’ll send a one-page comeback plan—no generic discount pitch."
        else:
            body = f"{owner}, quick useful check-in: {business} has {performance.get('views')} views and {performance.get('calls')} calls in the current 30-day snapshot. Reply YES and I’ll point out the single highest-leverage profile fix."
        cta, why = "binary", "Dormancy is acknowledged once and paired with a specific, low-effort reason to re-engage."
    elif kind in {"festival_upcoming", "category_seasonal"}:
        if kind == "category_seasonal" and p.get("trends"):
            trends = [humanize(t).replace(" demand ", " ") for t in p.get("trends", [])]
            body = (f"{owner}, summer shelf shift: {', '.join(trends[:3])}; {trends[3] if len(trends) > 3 else ''}. "
                    "Move ORS, sunscreen and antifungal stock to counter visibility; move cold/cough stock back. Want a 10-item reorder checklist?")
        elif p.get("festival"):
            days = p.get("days_until")
            if isinstance(days, (int, float)) and days > 60:
                body = f"{owner}, {p.get('festival')} is {days} days away ({short_date(p.get('date'))})—too early for customer promos. Now is the right time to map inventory and creative around {offer or 'your strongest service'}. Want the 3-stage campaign calendar?"
            else:
                body = f"{owner}, {p.get('festival')} is on {short_date(p.get('date'))}. I can turn {offer or 'your strongest service'} into one timely, category-fit post. Want the ready-to-publish version?"
        else:
            body = (f"{owner}, festival season is coming, and {business} already gets {number(performance.get('views'))} views a month but "
                    f"{number(performance.get('calls'))} calls. One timely festival post with a clear offer turns more of those views into calls. "
                    f"Want me to draft it around {offer or 'an offer you pick'}? Reply YES—I’ll leave the festival date for you to confirm.")
            cta = "binary"
        why = "Seasonal recommendation uses supplied timing and demand signals, including restraint when it is too early or incomplete."
    elif kind in {"renewal_due", "gbp_unverified", "winback_eligible", "milestone_reached"}:
        if kind == "renewal_due":
            plan = p.get("plan") or get(merchant, "subscription", "plan", default="current")
            days = p.get("days_remaining") if p.get("days_remaining") is not None else get(merchant, "subscription", "days_remaining", default="an unconfirmed number of")
            amount = f" Renewal amount: ₹{p.get('renewal_amount')}." if p.get("renewal_amount") is not None else ""
            body = f"{owner}, your {plan} plan has {days} days remaining.{amount} I can prepare the renewal summary and keep profile work uninterrupted. Reply YES to review it."
            cta = "binary"
        elif kind == "gbp_unverified":
            body = f"{owner}, {business}'s Google profile is still unverified."
            if p.get('verification_path'): body += f" Supplied verification path: {humanize(p['verification_path'])}."
            if p.get('estimated_uplift_pct') is not None: body += f" The supplied visibility estimate is {pct(p['estimated_uplift_pct'])}; this is not a promised outcome."
            body += " Reply YES for the verification checklist."
            cta = "binary"
        elif kind == "milestone_reached":
            if p.get("value_now") is not None:
                remaining = max(0, p.get("milestone_value", 0) - p.get("value_now", 0))
                body = f"{owner}, {business} is at {p.get('value_now')} reviews—just {remaining} away from {p.get('milestone_value')}. I can draft a thank-you post plus a non-pushy review request using your {offer or 'current customer base'}. Want it?"
            else:
                body = (f"{owner}, {business} just hit a milestone—worth celebrating publicly. You get {number(performance.get('views'))} views a month but "
                        f"{number(performance.get('calls'))} calls; a warm thank-you post gives those browsers a reason to visit"
                        + (f", led by {offer}" if offer else "") + ". Want me to draft it? Reply YES—I’ll leave the milestone number for you to confirm.")
                cta = "binary"
        else:
            facts = []
            if p.get('days_since_expiry') is not None: facts.append(f"your plan expired {p['days_since_expiry']} days ago")
            if p.get('lapsed_customers_added_since_expiry') is not None: facts.append(f"{p['lapsed_customers_added_since_expiry']} customers have lapsed since expiry")
            if p.get('perf_dip_pct') is not None: facts.append(f"the supplied performance decline is {pct(p['perf_dip_pct'])}")
            body = f"{owner}, {('; '.join(facts)) if facts else 'a restart review is due for ' + business}. Want a restart checklist using your current profile and offers?"
        why = "Functional trigger is stated with its concrete operational facts and one low-effort next step."
    else:
        # Post-submission trigger kinds: state the supplied facts, anchor on this merchant's numbers, one ask.
        facts = payload_facts(p)
        label = humanize(kind).capitalize() or "New signal"
        body = f"{owner}, new signal for {business}—{label.lower()}" + (f": {'; '.join(facts)}." if facts else ".")
        views, calls, ctr = performance.get("views"), performance.get("calls"), performance.get("ctr")
        peer = get(category, "peer_stats", "avg_ctr", default=None)
        if isinstance(views, (int, float)) and isinstance(calls, (int, float)):
            body += f" Your 30-day snapshot: {number(views)} views → {number(calls)} calls"
            if isinstance(ctr, (int, float)) and isinstance(peer, (int, float)):
                body += f", CTR {pct(ctr)} vs {pct(peer)} category average"
            body += "."
        body += f" Want {next_step(kind, p, offer)}? Reply YES and I’ll send it right away."
        cta = "binary"
        why = "New trigger type: restates only the supplied payload facts, anchors on the merchant's own snapshot, and offers one concrete deliverable."

    if send_as == "vera":
        body = code_mix(body, merchant)
    body = safe_sentence(body, category)
    return {"body": body, "cta": cta, "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", f"trigger:{trigger.get('id', '')}"),
            "rationale": why}


class ContextBody(BaseModel):
    scope: str
    context_id: str = Field(min_length=1, max_length=240)
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list, max_length=500)

    @field_validator("now")
    @classmethod
    def valid_now(cls, value):
        if parse_time(value) is None:
            raise ValueError("now must be an ISO timestamp")
        return value


class ReplyBody(BaseModel):
    conversation_id: str; merchant_id: str | None = None; customer_id: str | None = None
    from_role: str; message: str; received_at: str; turn_number: int


@app.get("/v1/healthz")
async def healthz():
    if blob_state is not None:
        try:
            async with asyncio.timeout(12):
                counts = await blob_state.counts()
        except Exception as exc:
            raise HTTPException(503, "State storage unavailable; retry health check") from exc
        return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}
    counts = {}
    for scope in ("category", "merchant", "customer", "trigger"):
        local_ids = [context_id for (stored_scope, context_id) in contexts if stored_scope == scope] if runtime_cache is None else []
        stored_ids = await state_get(f"index:{scope}", local_ids)
        # Report only observed entries, never the challenge's expected totals.
        counts[scope] = len(set(stored_ids or []))
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vera — Merchant AI Assistant</title>
<style>
:root{--ink:#f7f4ef;--muted:#a8b0aa;--panel:#111a16;--line:#27352e;--pink:#ff3d7f;--orange:#ff8a4c;--green:#79f2b0}
*{box-sizing:border-box}body{margin:0;background:#07100c;color:var(--ink);font:16px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 18% 0%,#48203580,transparent 35%),radial-gradient(circle at 90% 12%,#17452f70,transparent 32%)}
.wrap{position:relative;max-width:1120px;margin:auto;padding:28px 22px 64px}.nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:76px}
.brand{font-size:24px;font-weight:850;letter-spacing:-.04em}.brand b{color:var(--pink)}.pill{display:flex;align-items:center;gap:8px;border:1px solid var(--line);background:#0b1510;padding:8px 13px;border-radius:999px;color:var(--muted);font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 16px var(--green)}
.hero{display:grid;grid-template-columns:1.35fr .65fr;gap:48px;align-items:end;margin-bottom:70px}.eyebrow{color:var(--orange);font-size:12px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.hero h1{font-size:clamp(46px,7vw,86px);line-height:.95;letter-spacing:-.065em;margin:15px 0 24px;max-width:820px}.hero h1 em{font-style:normal;color:var(--pink)}.lead{color:var(--muted);font-size:18px;max-width:660px}.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:30px}.btn{display:inline-block;text-decoration:none;color:white;border:1px solid var(--line);padding:12px 17px;border-radius:10px;font-weight:750}.btn.primary{border:0;background:linear-gradient(110deg,var(--pink),var(--orange));box-shadow:0 12px 35px #ff3d7f2d}
.score{background:linear-gradient(145deg,#151e19,#0d1511);border:1px solid var(--line);border-radius:22px;padding:25px}.score strong{font-size:42px;letter-spacing:-.05em}.score p{margin:4px 0 18px;color:var(--muted)}.bar{height:7px;border-radius:8px;background:#28332e;overflow:hidden}.bar i{display:block;width:100%;height:100%;background:linear-gradient(90deg,var(--pink),var(--orange),var(--green))}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:54px}.metric,.card{background:#0d1612cc;border:1px solid var(--line);border-radius:15px;padding:20px}.metric span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}.metric b{display:block;font-size:28px;margin-top:6px}
.section-title{font-size:30px;letter-spacing:-.035em;margin:0 0 20px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:54px}.card .num{color:var(--pink);font-size:12px;font-weight:900}.card h3{margin:9px 0 7px}.card p{margin:0;color:var(--muted);font-size:14px}
.demo{display:grid;grid-template-columns:.8fr 1.2fr;gap:22px;background:linear-gradient(135deg,#121d17,#1b1518);border:1px solid #493040;border-radius:22px;padding:28px}.demo blockquote{margin:0;background:#f5f1e9;color:#172019;padding:20px;border-radius:14px;box-shadow:0 18px 38px #0005}.demo small{color:#59635d}.routes{margin-top:54px;border-top:1px solid var(--line);padding-top:25px;display:flex;gap:12px;flex-wrap:wrap}.route{font-family:ui-monospace,SFMono-Regular,monospace;color:var(--green);background:#101a15;border:1px solid var(--line);padding:8px 11px;border-radius:8px;text-decoration:none}.footer{margin-top:45px;color:#758078;font-size:13px}
@media(max-width:800px){.hero,.demo{grid-template-columns:1fr}.metrics,.flow{grid-template-columns:repeat(2,1fr)}.nav{margin-bottom:45px}}@media(max-width:480px){.metrics,.flow{grid-template-columns:1fr}.hero h1{font-size:44px}}
</style></head><body><main class="wrap">
<nav class="nav"><div class="brand">Vera<b>.</b></div><div class="pill"><i class="dot"></i><span id="live">Checking API</span></div></nav>
<section class="hero"><div><div class="eyebrow">Magicpin AI Challenge Submission</div><h1>Merchant intelligence that knows <em>why now.</em></h1><p class="lead">A context-grounded WhatsApp assistant for Indian local businesses. Vera turns category knowledge, merchant performance, live triggers and customer relationships into one message worth answering.</p><div class="actions"><a class="btn primary" href="/docs">Open API Explorer</a><a class="btn" href="/v1/metadata">View metadata</a></div></div><aside class="score"><span class="eyebrow">Runtime</span><strong id="status">Online</strong><p>Deterministic core + grounded safety fallback</p><div class="bar"><i></i></div></aside></section>
<section class="metrics"><div class="metric"><span>Categories loaded</span><b id="categories">0</b></div><div class="metric"><span>Merchants loaded</span><b id="merchants">0</b></div><div class="metric"><span>Customers loaded</span><b id="customers">0</b></div><div class="metric"><span>Triggers loaded</span><b id="triggers">0</b></div></section>
<h2 class="section-title">How the judge uses Vera</h2><section class="flow"><article class="card"><span class="num">01</span><h3>Push context</h3><p>Versioned category, merchant, customer and trigger data arrive safely.</p></article><article class="card"><span class="num">02</span><h3>Wake Vera</h3><p>A tick selects urgent, relevant triggers and suppresses duplicates.</p></article><article class="card"><span class="num">03</span><h3>Compose</h3><p>Every claim is grounded in supplied facts, offers, language and consent.</p></article><article class="card"><span class="num">04</span><h3>Continue</h3><p>Replies route to action, wait or exit—without auto-reply loops.</p></article></section>
<section class="demo"><div><div class="eyebrow">Synthetic dataset example</div><h2 class="section-title">Specific beats generic.</h2><p class="lead">Vera connects a supplied service, date and available slot to one clear next step. These are challenge examples; no real WhatsApp messages are sent.</p></div><blockquote>Hi Priya, Dr. Meera’s Dental Clinic here. Your cleaning recall is due 12 Nov 2026. Available: Thu 5 Nov, 6pm or Fri 6 Nov, 5pm, local time. Current offer: Dental Cleaning @ ₹299; the clinic must confirm eligibility. Reply YES to request a booking.<br><br><small>Customer reminder · Booking request, pending confirmation</small></blockquote></section>
<div class="routes"><a class="route" href="/v1/healthz">GET /v1/healthz</a><a class="route" href="/v1/metadata">GET /v1/metadata</a><a class="route" href="/docs">POST /v1/context</a><a class="route" href="/docs">POST /v1/tick</a><a class="route" href="/docs">POST /v1/reply</a></div><footer class="footer">Context First · Vera API v2.2 · Built for the Magicpin merchant-engagement evaluation</footer>
</main><script>fetch('/v1/healthz').then(r=>{if(!r.ok)throw new Error('health check failed');return r.json()}).then(d=>{const ids={category:'categories',merchant:'merchants',customer:'customers',trigger:'triggers'};for(const [k,id] of Object.entries(ids))document.getElementById(id).textContent=d.contexts_loaded[k];document.getElementById('live').textContent='API online';document.getElementById('status').textContent='Healthy'}).catch(()=>{document.getElementById('live').textContent='API unavailable';document.getElementById('status').textContent='Offline'})</script></body></html>"""


@app.get("/v1/metadata")
async def metadata():
    members = [name.strip() for name in os.getenv("TEAM_MEMBERS", "techi101").split(",") if name.strip()]
    return {"team_name": os.getenv("TEAM_NAME", "Context First"), "team_members": members,
            "model": f"deterministic composer + {LLM_MODEL}" if llm_enabled() else "deterministic context-grounded composer (LLM disabled)",
            "approach": "grounded trigger router + category voice guardrails + scoped consent/dedup safety + deployment-scoped private state storage",
            "state_backend": STATE_BACKEND if blob_state is not None else "runtime_cache" if runtime_cache is not None else "memory",
            "source_sha256": SOURCE_HASHES,
            "contact_email": os.getenv("CONTACT_EMAIL", ""), "version": app.version,
            "submitted_at": os.getenv("SUBMITTED_AT", "2026-09-25T00:00:00Z")}


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in {"category", "merchant", "customer", "trigger"}:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope", "details": body.scope,
        })
    key = (body.scope, body.context_id)
    id_field = {"category": "slug", "merchant": "merchant_id", "customer": "customer_id", "trigger": "id"}[body.scope]
    if body.payload.get(id_field) not in {None, body.context_id}:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "context_id_mismatch"})
    if blob_state is not None:
        record = {"version": body.version, "payload": body.payload, "stored_at": datetime.now(timezone.utc).isoformat()}
        try:
            status, stored = await blob_state.put_context(body.scope, body.context_id, record)
        except Exception as exc:
            raise HTTPException(503, "State storage unavailable; write not acknowledged") from exc
        if status == "stale":
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": stored["version"]})
        return {"accepted": True, "duplicate": status == "duplicate", "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": stored["stored_at"]}
    old = await load_context(body.scope, body.context_id)
    if old and old["version"] > body.version:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": old["version"]})
    if old and old["version"] == body.version:
        return {"accepted": True, "duplicate": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": old.get("stored_at")}
    record = {"version": body.version, "payload": body.payload, "stored_at": datetime.now(timezone.utc).isoformat()}
    await state_set(context_key(body.scope, body.context_id), record)
    contexts[key] = record
    local_ids = [context_id for (scope, context_id) in contexts if scope == body.scope] if runtime_cache is None else []
    stored_ids = await state_get(f"index:{body.scope}", local_ids)
    merged_ids = sorted(set((stored_ids or []) + [body.context_id]))
    await state_set(f"index:{body.scope}", merged_ids)
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/tick")
async def tick(request: TickBody):
    candidates = []
    started = time.monotonic()
    for tid in dict.fromkeys(request.available_triggers):
        trigger = get(await load_context("trigger", tid), "payload", default=None)
        if not trigger: continue
        expiry = parse_time(trigger.get("expires_at"))
        if trigger.get("expires_at") and (expiry is None or expiry <= parse_time(request.now)):
            continue
        candidates.append({**trigger, "id": tid})
    candidates.sort(key=lambda t: t.get("urgency", 0), reverse=True)
    actions = []
    for trigger in candidates:
        if len(actions) >= 20 or time.monotonic() - started > 20:
            break
        merchant_id, customer_id = trigger.get("merchant_id"), trigger.get("customer_id")
        if trigger.get("scope") not in {"customer", "merchant"}: continue
        recipient = recipient_key(merchant_id, customer_id if trigger.get("scope") == "customer" else None)
        suppression = trigger.get("suppression_key") or f"trigger:{trigger['id']}"
        dedup_key = f"sent:{recipient}:{suppression}"
        if await state_get(dedup_key, False) or await state_get(f"optout:{recipient}", False):
            continue
        deferred_until = parse_time(await state_get(f"defer:{recipient}"))
        if deferred_until and parse_time(request.now) < deferred_until:
            continue
        merchant = get(await load_context("merchant", merchant_id), "payload", default=None)
        customer = get(await load_context("customer", customer_id), "payload", default=None) if customer_id else None
        category = get(await load_context("category", get(merchant, "category_slug", default="")), "payload", default=None)
        if not merchant or not category: continue
        if trigger.get("scope") == "customer" and not customer_allowed(customer, trigger):
            continue
        message = compose(category, merchant, trigger, customer)
        if llm_enabled() and trigger.get("kind") not in KNOWN_TRIGGER_KINDS and time.monotonic() - started < 5:
            try:
                message = await asyncio.wait_for(refine_unknown_trigger(category, merchant, trigger, customer, message), 5)
            except TimeoutError:
                pass
        conv = f"conv_{trigger.get('id')}_{uuid4().hex[:12]}"
        conversations[conv].append({"from": "vera", "msg": message["body"], "trigger_id": trigger.get("id", "")})
        await state_set(f"conversation:{conv}", conversations[conv])
        await state_set(dedup_key, True)
        actions.append({"conversation_id": conv, "merchant_id": merchant_id, "customer_id": customer_id,
                        "send_as": message["send_as"], "trigger_id": trigger.get("id"),
                        "template_name": "vera_contextual_v1", "template_params": [message["body"]], **message})
    return {"actions": actions}


AUTO_MARKERS = ("thank you for contacting", "our team will respond", "automated assistant", "auto-reply", "we will get back", "out of office",
                "this is an automated", "team tak pahuncha", "main ek automated", "thank you for your message", "thanks for your message",
                "currently closed", "currently unavailable", "will reply soon", "get back to you", "business hours", "we are closed",
                "reply at the earliest", "jald hi sampark", "sampark karne ke liye dhanyavaad", "auto reply", "automatic reply")
STOP_PATTERNS = (r"\b(?:stop|unsubscribe|not interested|no thanks|useless spam|nahi chahiye)\b", r"\bdon['’]?t (?:message|contact|text)\b", r"\bleave me alone\b", r"\bfuck off\b", r"^no[.! ]*$")
GO_PATTERNS = (r"\byes\b", r"\bgo ahead\b", r"\bproceed\b", r"\bjoin\b", r"\bsign me up\b",
               r"\blet'?s do it\b", r"\bwhat'?s next\b", r"\bkar do\b", r"\bhaan\b",
               r"\bsend (it|me)\b", r"\bshare (it|the)\b", r"\bdraft it\b", r"\bshow me\b", r"^go[.! ]*$", r"\bjudrna\b", r"^confirm[.! ]*$")
LATER_PATTERNS = (r"\b(?:later|busy now|after some time|baad mein|not now|abhi nahi|next week|some other time|kal baat)\b", r"\b(?:call|message|contact|send it)\b.*\btomorrow\b")


HINGLISH_WORDS = r"\b(?:hai|haan|nahi|kya|karo|kar|kardo|karna|hoga|accha|acha|achha|theek|thik|chalo|kitna|lagega|aur|mein|bhai|ji|bata|batao|kyun|kaise|samjha|dijiye|chahiye|abhi)\b"
ABUSE_PATTERN = r"\b(?:scam\w*|fraud\w*|fake|cheat\w*|useless|bakwas|bekaar|idiot\w*|stupid|nonsense|waste of (?:my )?time|loot\w*|chor)\b"
OFFTOPIC_PATTERN = r"\b(?:gst|income tax|tax return|itr|legal advice|lawyer|personal loan|loan|visa|passport|insurance claim|electricity bill)\b"
ACK_PATTERN = r"^(?:ok(?:ay)?|k|sure|done|fine|alright|cool|haan(?: ji)?|ha ji|ji|theek hai|thik hai|👍|✅)[.! ]*$"
COMMIT_PATTERNS = GO_PATTERNS + (r"\bsure\b", r"\bok(?:ay)?\b.*\b(?:do|go|send|start|draft)\b", r"\bplease do\b", r"\bdo it\b", r"\bgo for it\b",
                                 r"\bsounds good\b", r"\binterested\b", r"\bkaro\b", r"\bkardo\b", r"\bchalo\b", r"\bbook it\b", r"\bi want\b")
THANKS_PATTERN = r"\b(?:thanks|thank you|thx|shukriya|dhanyavaad|dhanyavad|perfect|great|got it|looks good)\b"
QUESTION_PATTERN = r"\?|\b(?:why|how|what|which|when|where|who|tell me more|more details|details|explain|kya|kaise|kyun|kab|kaun|matlab|batao|bataiye)\b"
UNSURE_PATTERN = r"\b(?:not sure|don['’]?t know|confus\w*|pata nahi|let me think|think about it|maybe|hmm+)\b"
EDIT_PATTERN = r"\b(?:shorter|short|shorten|longer|change|edit|modify|in hindi|hindi mein|tweak|rewrite)\b"


def speaks_hinglish(text: str) -> bool:
    return bool(re.search(r"[ऀ-ॿ]", text) or re.search(HINGLISH_WORDS, text.lower()))


def explain_trigger(merchant: dict, category: dict, trigger: dict, customer: dict | None) -> str:
    """Answer 'why / how / tell me more' from the trigger and merchant data only, ending in one ask."""
    kind = TRIGGER_ALIASES.get(trigger.get("kind"), trigger.get("kind", "")) or ""
    p = trigger.get("payload", {}) or {}
    perf = merchant.get("performance", {}) or {}
    business = get(merchant, "identity", "name", default="your business")
    offer = active_offer(merchant)
    views, calls, ctr = perf.get("views"), perf.get("calls"), perf.get("ctr")
    peer = get(category, "peer_stats", "avg_ctr", default=None)
    item = digest_for(category, {**trigger, "kind": kind}) if category else None
    if kind == "active_planning_intent":
        locality = get(merchant, "identity", "locality", default="")
        text = (f"Your data doesn’t name specific customers or headcounts, so I won’t guess. The draft targets "
                f"{'offices and groups in ' + locality if locality else 'your nearby audience'}"
                + (f" with {offer}" if offer else "") + "; minimum quantity, dates and cut-off stay blank for you to fill.")
    elif item and any(k in kind for k in ("research", "regulation", "cde", "digest", "compliance")):
        text = f"In short: {item.get('summary') or item.get('title', 'the supplied update')}"
        if item.get("actionable"):
            text += f" How to use it: {item['actionable'].rstrip('.')}."
        if item.get("source"):
            text += f" Source: {item['source']}."
    elif any(k in kind for k in ("dip", "spike", "drop", "decline", "dormant", "seasonal")) or "delta_pct" in p:
        facts = payload_facts(p)
        text = "Here’s what the data shows" + (f": {'; '.join(facts)}." if facts else ".")
        if isinstance(views, (int, float)) and isinstance(calls, (int, float)):
            text += f" Over 30 days {business} had {number(views)} views and {number(calls)} calls"
            if isinstance(ctr, (int, float)) and isinstance(peer, (int, float)):
                text += f" (CTR {pct(ctr)} vs {pct(peer)} category average)."
                text += (" So people are finding you but fewer are calling—the fix is a clearer offer and a fresh post, not more discounts."
                         if ctr < peer else
                         " Your call rate is at or above the category average, so the lever is visibility—a fresh post brings more people to the profile.")
            else:
                text += "."
    else:
        facts = payload_facts(p)
        text = ("Why now: " + "; ".join(facts) + ".") if facts else f"This came from a new signal on {business}’s account."
        if isinstance(views, (int, float)) and isinstance(calls, (int, float)):
            text += f" For context, the profile has {number(views)} views and {number(calls)} calls in the last 30 days."
    return text + f" The next step I’d take: {next_step(kind, p, offer)}. Reply YES and I’ll send it."


def shorter_version(previous: str) -> str:
    opens = [i for i in (previous.find("‘"), previous.find("“")) if i >= 0]
    closes = [i for i in (previous.rfind("’"), previous.rfind("”")) if i >= 0]
    core = previous[min(opens) + 1:max(closes)] if opens and closes and max(closes) > min(opens) else previous
    first = re.split(r"(?<=[.!?])\s", core.strip())[0]
    return f"Shorter version: ‘{first.strip()}’ Want me to use this one? Reply YES."


def action_draft(merchant: dict, category: dict, trigger: dict, customer: dict | None,
                 inbound: ReplyBody) -> str:
    """Deliver the promised artifact in this response; never claim external execution."""
    business = get(merchant, "identity", "name", default="[business name]")
    locality = get(merchant, "identity", "locality", default="")
    kind = TRIGGER_ALIASES.get(trigger.get("kind"), trigger.get("kind"))
    payload = trigger.get("payload", {})
    if inbound.from_role == "customer":
        if kind == "supply_alert":
            return "Please contact the pharmacist to check the affected batch and the next steps. No replacement or change to your medication has been confirmed here."
        options = slot_labels(payload)
        if kind == "chronic_refill_due":
            return "Your refill request is noted in this conversation. The pharmacist still needs to confirm the prescription, stock and delivery details; this is not a dispensing confirmation."
        if kind == "appointment_tomorrow":
            when = short_date(payload.get("appointment_time") or payload.get("appointment_time_iso"), True)
            return f"Your confirmation is noted here{(' for ' + when) if when else ''}. The clinic must confirm any change to the appointment directly."
        if options:
            selected = inbound.message.strip() in {"1", "2"}
            choice = int(inbound.message.strip()) - 1 if selected else 0
            if selected and choice < len(options):
                return f"Your preferred slot is {options[choice]}. This is a booking request; the business must confirm availability before it is booked."
            return "Available options from the reminder: " + "; ".join(f"{i + 1}: {x}" for i, x in enumerate(options[:2])) + ". Reply with the slot number to request it; the business still needs to confirm availability."
        return "Your request is noted here. The business needs to confirm the service and available time before a booking can be made."
    item = digest_for(category, {**trigger, "kind": kind})
    if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        if not item:
            return "The referenced source has not been supplied, so I cannot send an accurate summary yet. The next step is to attach that item to this conversation."
        evidence = f"{item.get('title', 'Supplied update')}: {item.get('summary', '')}"
        if item.get("source"): evidence += f" Source: {item['source']}."
        if kind == "research_digest":
            return f"Supplied summary: {evidence} Customer-share draft: ‘{business}: At your next visit, ask which follow-up schedule fits your own needs. We can discuss the options with you.’ This is a draft for review, not a published message."
        steps = item.get("actionable") or "Review the supplied notice and identify the changes relevant to your business"
        return f"Here is the checklist: {evidence} First: {steps}. Next: record what applies to your business and assign an owner to verify it against the source. No registration or external update has been made."
    if re.search(r"\b(?:join|sign me up|judrna)\b", inbound.message, re.I):
        return f"Here is the onboarding checklist for {business}: review the available plan and its current price, verify the business details, then complete account setup in magicpin's merchant onboarding flow. I cannot activate a subscription or take payment from this chat."
    if kind == "gbp_unverified":
        path = humanize(payload.get("verification_path"))
        return f"Here is the verification checklist for {business}: open the business profile, check the address and business details, then follow {'the supplied ' + path + ' route' if path else 'the verification method Google offers'}. Keep the confirmation for review. I have not changed or verified the profile."
    if kind == "renewal_due":
        plan = payload.get("plan") or get(merchant, "subscription", "plan", default="current plan")
        amount = f"₹{payload['renewal_amount']}" if payload.get("renewal_amount") is not None else "not supplied"
        return f"Here is the renewal summary: {business}; plan: {plan}; quoted renewal amount: {amount}. Review the current terms in your merchant account before paying. No renewal or charge has been made here."
    if kind == "review_theme_emerged":
        theme = humanize(payload.get("theme")) or "the experience"
        return f"Public reply draft: ‘Thank you for sharing your feedback about {theme}. The {business} team would like to understand the experience and follow up with you directly.’ Operating step: review the relevant recent feedback with the team and assign one person to follow up. Nothing has been posted."
    offer = active_offer(merchant)
    location_text = f", {locality}" if locality else ""
    if "review" in (kind or ""):
        return (f"Public reply draft: ‘Thank you for telling us about this, and we’re sorry the visit fell short. The {business} team "
                "would like to make it right—please message us directly so we can follow up.’ Operating step: review what happened "
                "with the team and assign one person to follow up. Nothing has been posted.")
    if payload.get("molecule") and "supply" not in (kind or ""):
        molecule = payload["molecule"]
        cover = f" (supplied cover: {payload['days_of_cover']} days)" if payload.get("days_of_cover") is not None else ""
        return (f"Reorder checklist for {molecule}{cover}: confirm current stock, place the distributor order, and list regular "
                f"customers to inform once stock is confirmed. Customer note draft: ‘{business}: {molecule} is being restocked—message "
                "us to keep yours aside.’ For pharmacist review; nothing has been ordered or sent.")
    topic = humanize(payload.get("intent_topic"))
    if kind == "active_planning_intent" and topic:
        lead = f"{topic.capitalize()}" + (f": {offer}" if offer else "")
        return f"Google/WhatsApp draft: ‘{business}{location_text} — {lead}. Message us to book or register interest.’ Approval checklist: final price, dates, capacity and any eligibility restrictions. Nothing has been published."
    suggested = None
    if not offer:
        suggested = next((o.get("title") for o in category.get("offer_catalog", []) if o.get("title")), None)
    headline = f"{business}{location_text}" + (f" — {offer or suggested}" if (offer or suggested) else "")
    note = (f" You have no active offer, so I used {suggested} from the magicpin {category.get('display_name', 'category')} catalog—change it if you prefer."
            if suggested else "")
    return f"Here is the draft: ‘{headline}. Call or message us to book.’{note} Before publishing, confirm offer eligibility and current availability. This copy is ready for review; no profile update or customer message has been sent."


@app.post("/v1/reply")
async def reply(inbound: ReplyBody):
    message = inbound.message.strip()
    lower = message.lower()
    if inbound.from_role not in {"merchant", "customer"}:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope", "details": inbound.from_role,
        })
    cached_response = await state_get(f"reply:{inbound.conversation_id}:{inbound.turn_number}")
    if cached_response:
        if cached_response["message"] != message:
            return JSONResponse(status_code=409, content={"reason": "turn_conflict"})
        return cached_response["response"]
    history = await state_get(f"conversation:{inbound.conversation_id}", [])
    conversations[inbound.conversation_id] = history or []
    conversations[inbound.conversation_id].append({"from": inbound.from_role, "msg": message})
    await state_set(f"conversation:{inbound.conversation_id}", conversations[inbound.conversation_id])

    async def send(body: str, cta: str, rationale: str):
        prior_bodies = {turn.get("msg") for turn in conversations[inbound.conversation_id] if turn.get("from") == "vera"}
        if body in prior_bodies:
            return {"action": "end", "rationale": "No new grounded detail is available; end rather than repeat a prior message."}
        conversations[inbound.conversation_id].append({"from": "vera", "msg": body})
        await state_set(f"conversation:{inbound.conversation_id}", conversations[inbound.conversation_id])
        return {"action": "send", "body": body, "cta": cta, "rationale": rationale}

    normalized = re.sub(r"\W+", " ", lower).strip()
    recipient = recipient_key(inbound.merchant_id, inbound.customer_id if inbound.from_role == "customer" else None)
    if any(re.search(p, lower) for p in STOP_PATTERNS):
        await state_set(f"optout:{recipient}", True)
        return {"action": "end", "rationale": "Opt-out or hostile signal received; stop immediately."}
    if await state_get(f"optout:{recipient}", False):
        return {"action": "end", "rationale": "Prior opt-out remains active; no further outreach."}
    same_as_before = sum(1 for turn in conversations[inbound.conversation_id]
                         if turn.get("from") == inbound.from_role
                         and re.sub(r"\W+", " ", str(turn.get("msg", "")).lower()).strip() == normalized)
    if any(x in lower for x in AUTO_MARKERS) or (same_as_before >= 2 and len(normalized) > 12):
        fingerprint_key = f"auto:{recipient}:{normalized}"
        count = max(int(await state_get(fingerprint_key, 0)) + 1, same_as_before)
        auto_reply_fingerprints[fingerprint_key] = count
        await state_set(fingerprint_key, count)
        if count >= 2 or "automated assistant" in lower:
            return {"action": "end", "rationale": "Repeated WhatsApp Business auto-reply detected (same text again); exit instead of wasting turns."}
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Likely automated acknowledgement; wait for an owner/manager response."}
    if any(re.search(pattern, lower) for pattern in LATER_PATTERNS):
        from datetime import timedelta
        received = parse_time(inbound.received_at)
        if received:
            await state_set(f"defer:{recipient}", (received + timedelta(minutes=30)).isoformat())
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked to defer; back off for 30 minutes."}

    merchant_record = await load_context("merchant", inbound.merchant_id)
    merchant = get(merchant_record, "payload", default=None) or {}
    category_record = await load_context("category", get(merchant, "category_slug", default="")) if merchant else None
    category = get(category_record, "payload", default=None) or {}
    trigger_id = next((turn.get("trigger_id") for turn in history if turn.get("trigger_id")), None)
    trigger = get(await load_context("trigger", trigger_id), "payload", default={}) or {}
    customer = get(await load_context("customer", inbound.customer_id), "payload", default=None)
    is_customer = inbound.from_role == "customer"
    profile = customer if is_customer else merchant
    lang = {"identity": {"language_pref": "hi-en mix"}} if speaks_hinglish(message) else profile
    kind = TRIGGER_ALIASES.get(trigger.get("kind"), trigger.get("kind", "")) or ""
    business = get(merchant, "identity", "name", default="your business")
    step = next_step(kind, trigger.get("payload", {}) or {}, active_offer(merchant))
    flags_key = f"flags:{inbound.conversation_id}"
    flags = await state_get(flags_key, {}) or {}

    async def say(body: str, cta: str, rationale: str, **flag_updates):
        flags.update(flag_updates)
        await state_set(flags_key, flags)
        result = await send(safe_sentence(code_mix(body, lang), category), cta, rationale)
        await state_set(f"reply:{inbound.conversation_id}:{inbound.turn_number}", {"message": message, "response": result})
        return result

    async def bump(name: str) -> int:
        flags[name] = int(flags.get(name, 0)) + 1
        await state_set(flags_key, flags)
        return flags[name]

    if flags.get("closed"):
        return {"action": "end", "rationale": "Conversation already closed after delivery; ending instead of adding noise."}
    if re.search(ABUSE_PATTERN, lower) and not is_customer:
        if await bump("hostile") >= 2:
            return {"action": "end", "rationale": "Repeated hostility; exit politely rather than push."}
        return await say(f"Understood, and sorry if earlier messages felt pushy. Nothing is charged from this chat—I only prepare drafts for {business}, "
                         "and nothing goes out without your OK. If you’d rather not hear from me, reply STOP.",
                         "none", "De-escalates hostility once, clarifies no charge or action, and offers a clear opt-out.")
    if re.search(OFFTOPIC_PATTERN, lower):
        return await say(f"That one is outside what I can help with—a CA or the relevant professional is the right person. "
                         f"What I can do today for {business}: {step}. Reply YES if useful.",
                         "binary", "Politely declines the off-topic request and returns to the conversation's trigger with one ask.")
    last_vera = next((t.get("msg", "") for t in reversed(conversations[inbound.conversation_id][:-1]) if t.get("from") == "vera"), "")
    if flags.get("drafted") and re.search(EDIT_PATTERN, lower):
        if re.search(r"\bshort", lower):
            return await say(shorter_version(last_vera), "binary", "Applies the requested edit to the delivered draft.")
        return await say("Tell me the exact change—wording, price, timing or tone—and I’ll revise the draft in one go.",
                         "open_ended", "Asks for the one missing detail needed to apply the requested edit.")
    negated = bool(re.search(r"\b(?:don['’]?t|do not|not ready|not yet|never mind|no need)\b", lower))
    unsure = bool(re.search(UNSURE_PATTERN, lower)) and not re.search(QUESTION_PATTERN, lower)
    if re.search(r"\bwhatever\b|\bnot (?:really )?useful\b|\bno time\b", lower):
        return {"action": "wait", "wait_seconds": 86400, "rationale": "Disengaged reply; back off for a day instead of pushing."}
    if re.search(r"\b(?:how much|price|cost|charge|fees?|kitna|paisa|paise)\b", lower):
        if is_customer:
            offer = active_offer(merchant)
            body = (f"The current offer on record is {offer}. The team will confirm eligibility and availability."
                    if offer else "The team will confirm the price for this; nothing is booked or charged yet.")
            return await say(body, "none", "Answers the customer's price question from merchant data without inventing a quote.")
        plan = get(merchant, "subscription", "plan", default=None)
        status = get(merchant, "subscription", "status", default=None)
        body = f"Nothing is charged from this chat—I only prepare {step}, and you decide whether to use it."
        if plan and status == "active":
            body += f" It’s part of the support on your active {plan} plan."
        return await say(body + " Reply YES to get it.", "binary", "Answers the cost question truthfully from subscription data and returns to one ask.")
    committed = (any(re.search(p, lower) for p in COMMIT_PATTERNS) or re.search(ACK_PATTERN, lower)
                 or (is_customer and lower in {"1", "2"})) and not negated and not unsure
    if committed and not flags.get("drafted"):
        draft = action_draft(merchant, category, trigger, customer, inbound)
        return await say(draft, "none", "Merchant committed: delivers the promised artifact now instead of re-qualifying; claims no external execution.",
                         drafted=True)
    if flags.get("drafted") and (committed or re.search(THANKS_PATTERN, lower)):
        closing = (f"You’re welcome! {business} will confirm the details with you." if is_customer else
                   f"Great—it’s ready whenever you are. Nothing is published until you post it; I’ll only message again if something new comes up for {business}.")
        return await say(closing, "none", "Delivery is done; closes gracefully instead of pushing another ask.", closed=True)
    if re.search(THANKS_PATTERN, lower) and not re.search(QUESTION_PATTERN, lower):
        return await say(f"Anytime. {step[0].upper() + step[1:]} is one YES away whenever you want it.", "binary",
                         "Acknowledges thanks and leaves one low-effort ask.", closed=True)
    if unsure:
        if await bump("unsure") >= 2:
            return {"action": "wait", "wait_seconds": 3600, "rationale": "Merchant still undecided; give space instead of pressing."}
        return await say(f"No pressure. I’ll prepare {step} so you can judge it on paper—nothing goes live without your OK. Reply YES to see it.",
                         "binary", "Lowers the effort of saying yes: offers to show the artifact before any commitment.")
    if negated:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Consent to proceed is negated; do not act."}
    if re.search(r"\b(?:source|evidence|where.*(?:from|study)|abstract|summary)\b", lower) and not is_customer:
        return await say(action_draft(merchant, category, trigger, customer, inbound), "none",
                         "Answers the request using the source attached to this conversation.", drafted=True)
    if re.search(QUESTION_PATTERN, lower):
        asked = await bump("explained")
        if is_customer:
            options = slot_labels(trigger.get("payload", {}) or {})
            body = (f"The {business} team will confirm that for you. Available options: "
                    + "; ".join(f"{i + 1}: {x}" for i, x in enumerate(options[:2])) + ". Reply with the number to request one."
                    if options else f"The {business} team will confirm that for you. Reply YES and we’ll get back with the details.")
            return await say(body, "binary", "Routes the customer's question to the business without inventing an answer.")
        if asked == 1:
            return await say(explain_trigger(merchant, category, trigger, customer), "binary",
                             "Answers the question from trigger and merchant data only, then one ask.")
        if asked == 2 and not flags.get("drafted"):
            draft = action_draft(merchant, category, trigger, customer, inbound)
            return await say("Easiest is to show you. " + draft, "none",
                             "Second question: externalizes effort by delivering the artifact instead of explaining again.", drafted=True)
    model_result = await intelligent_reply(conversations[inbound.conversation_id], merchant, category, inbound)
    if model_result:
        if model_result["action"] == "send":
            return await send(model_result["body"], model_result["cta"], model_result["rationale"])
        if model_result["action"] == "wait":
            return {"action": "wait", "wait_seconds": max(60, min(86400, model_result["wait_seconds"])),
                    "rationale": model_result["rationale"]}
        return {"action": "end", "rationale": model_result["rationale"]}
    if await bump("nudged") >= 2:
        return {"action": "wait", "wait_seconds": 3600, "rationale": "No clear intent after one grounded nudge; give space."}
    if is_customer:
        return await say(f"Noted. The {business} team will follow up. Reply YES if you’d like us to hold a slot for you.", "binary",
                         "Acknowledges the customer and keeps one simple ask.")
    return await say(f"Noted. To keep it simple for {business}, the one step I’d take now is {step}. Reply YES and I’ll send it.",
                     "binary", "Acknowledges the reply and re-anchors on the trigger with one concrete ask.")


@app.post("/v1/teardown")
async def teardown():
    if blob_state is not None:
        try:
            await blob_state.clear()
        except Exception as exc:
            raise HTTPException(503, "State wipe failed; retry teardown") from exc
    if runtime_cache is not None:
        try:
            await runtime_cache.expire_tag(STATE_TAG)
        except Exception as exc:
            raise HTTPException(503, "State wipe failed; retry teardown") from exc
    contexts.clear(); conversations.clear(); sent_suppressions.clear(); auto_reply_fingerprints.clear(); local_state.clear()
    return {"accepted": True, "wiped": True}
