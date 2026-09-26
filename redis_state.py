"""Deployment-scoped challenge state on Upstash Redis (Vercel Marketplace, free tier).

Replaces Vercel Blob as the primary store: Hobby Blob allows only 2,000 put/list
operations per 30 days, fewer than one judge run needs. Redis gives atomic
version replacement in a single Lua call and O(1) counts from per-scope sets,
so no listing is needed for /v1/healthz.
"""
import asyncio
import json
import re

import httpx

# KEYS[1]=context key, KEYS[2]=scope id set; ARGV[1]=record JSON, ARGV[2]=version, ARGV[3]=context id.
PUT_CONTEXT_LUA = """
local current = redis.call('GET', KEYS[1])
if current then
  local old = cjson.decode(current)['version']
  local new = tonumber(ARGV[2])
  if old >= new then
    if old == new then return {'duplicate', current} end
    return {'stale', current}
  end
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], ARGV[3])
return {'accepted', ARGV[1]}
"""
SCOPES = ("category", "merchant", "customer", "trigger")


def transient(exc):
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class RedisState:
    def __init__(self, namespace, url, token):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", namespace):
            raise ValueError("A nonempty, safe challenge namespace is required")
        if not url or not token:
            raise ValueError("Upstash REST URL and token are required")
        self.prefix = namespace + ":"
        self._url = url.rstrip("/")
        self._token = token
        self._client = None

    @property
    def _http(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=8, headers={"Authorization": f"Bearer {self._token}"})
        return self._client

    async def command(self, *args):
        # EVAL and SET are idempotent for our usage, so transient failures retry safely:
        # a replayed context write resolves as "duplicate".
        for failures in range(4):
            try:
                response = await self._http.post(self._url, json=[str(a) for a in args])
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise RuntimeError(f"Redis error: {body['error']}")
                return body.get("result")
            except Exception as exc:
                if not transient(exc) or failures == 3:
                    raise
                await asyncio.sleep(0.25 * 2 ** failures)

    def context_key(self, scope, context_id):
        if scope not in SCOPES:
            raise ValueError("Unknown context scope")
        return f"{self.prefix}context:{scope}:{context_id}"

    def state_key(self, key):
        if key.startswith("context:"):
            _, scope, context_id = key.split(":", 2)
            return self.context_key(scope, context_id)
        return f"{self.prefix}state:{key}"

    async def get(self, key):
        raw = await self.command("GET", self.state_key(key))
        return None if raw is None else json.loads(raw)

    async def set(self, key, value, options=None):
        await self.command("SET", self.state_key(key), json.dumps(value))

    async def put_context(self, scope, context_id, record):
        status, stored = await self.command(
            "EVAL", PUT_CONTEXT_LUA, 2, self.context_key(scope, context_id), f"{self.prefix}ids:{scope}",
            json.dumps(record), int(record["version"]), context_id)
        return status, json.loads(stored)

    async def counts(self):
        result = await self.command(
            "EVAL", "local r = {} for i, k in ipairs(KEYS) do r[i] = redis.call('SCARD', k) end return r",
            len(SCOPES), *(f"{self.prefix}ids:{scope}" for scope in SCOPES))
        return dict(zip(SCOPES, (int(n) for n in result)))

    async def clear(self):
        # Only this deployment's namespace; never FLUSHDB a shared database.
        cursor = "0"
        while True:
            cursor, keys = await self.command("SCAN", cursor, "MATCH", self.prefix + "*", "COUNT", 1000)
            if any(not key.startswith(self.prefix) for key in keys):
                raise RuntimeError("Storage returned an out-of-namespace key")
            if keys:
                await self.command("DEL", *keys)
            if str(cursor) == "0":
                break

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
