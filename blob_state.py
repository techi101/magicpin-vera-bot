"""Private, consistently-read challenge state on Vercel Blob.

One context per object eliminates shared read/modify/write indexes. ETag
preconditions make version replacement atomic across separate Function workers.
The Python SDK lacks conditional put, so that call uses the official Blob REST
headers documented by vercel/storage's put-helpers.ts (x-if-match).
"""
import asyncio
import hashlib
import json
import re

import httpx
from vercel.blob import AsyncBlobClient


class StorageConflict(Exception):
    pass


def transient(exc):
    """Upstream blips (5xx, 429, timeouts) observed at ~2% of warmup writes."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class PrivateBlobState:
    def __init__(self, namespace, token):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", namespace):
            raise ValueError("A nonempty, safe challenge namespace is required")
        self.prefix = namespace + "/"
        self._token = token
        token_match = re.match(r"^vercel_blob_rw_([A-Za-z0-9]+)_", token)
        if not token_match:
            raise ValueError("A Vercel Blob read/write token is required")
        self._origin = f"https://{token_match.group(1).lower()}.private.blob.vercel-storage.com"
        self._blobs = AsyncBlobClient(token=token)
        self._http = httpx.AsyncClient(timeout=12)

    def path(self, key):
        digest = hashlib.sha256(key.encode()).hexdigest()
        if key.startswith("context:"):
            scope = key.split(":", 2)[1]
            if scope not in {"category", "merchant", "customer", "trigger"}:
                raise ValueError("Unknown context scope")
            return f"{self.prefix}context/{scope}/{digest}.json"
        return f"{self.prefix}state/{digest}.json"

    async def read_record(self, key):
        # CDN compression changes a large object's ETag to W/"...", which is
        # invalid for If-Match. Request the identity representation so the
        # strong ETag describes exactly the stored bytes used by conditional PUT.
        response = await self._http.get(f"{self._origin}/{self.path(key)}", params={"cache": "0"},
                                        headers={"Authorization": f"Bearer {self._token}", "Accept-Encoding": "identity"})
        if response.status_code == 404:
            return None, None
        response.raise_for_status()
        etag = response.headers.get("etag")
        if not etag or etag.startswith("W/"):
            raise RuntimeError("Storage did not supply a strong version ETag")
        return response.json(), etag

    async def get(self, key):
        value, _ = await self.read_record(key)
        return value

    async def set(self, key, value, options=None):
        await self._blobs.put(self.path(key), json.dumps(value).encode(),
                              access="private", overwrite=True,
                              content_type="application/json", add_random_suffix=False)

    async def conditional_put(self, key, value, etag):
        headers = {"Authorization": f"Bearer {self._token}", "x-api-version": "12",
                   "x-vercel-blob-access": "private", "x-add-random-suffix": "0",
                   "x-content-type": "application/json", "x-allow-overwrite": "1" if etag else "0"}
        if etag:
            headers["x-if-match"] = etag
        response = await self._http.put("https://vercel.com/api/blob/",
                                        params={"pathname": self.path(key)}, headers=headers,
                                        content=json.dumps(value).encode())
        if response.is_error:
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            if (response.status_code == 412 or error.get("code") == "precondition_failed"
                    or (not etag and error.get("code") == "bad_request"
                        and "already exists" in error.get("message", "").lower())):
                raise StorageConflict()
            response.raise_for_status()

    async def put_context(self, scope, context_id, record):
        key = f"context:{scope}:{context_id}"
        # A retried write that had actually landed resolves as a duplicate on
        # the next read, so retrying transient failures is idempotent.
        failures = 0
        for attempt in range(8):
            try:
                old, etag = await self.read_record(key)
                if old and old["version"] >= record["version"]:
                    return ("duplicate" if old["version"] == record["version"] else "stale"), old
                await self.conditional_put(key, record, etag)
                return "accepted", record
            except StorageConflict:
                await asyncio.sleep(0.01 * (attempt + 1))
            except Exception as exc:
                failures += 1
                if not transient(exc) or failures > 3:
                    raise
                await asyncio.sleep(0.25 * 2 ** (failures - 1))
        raise RuntimeError("Concurrent context update did not settle; retry")

    async def objects(self, suffix=""):
        cursor = None
        prefix = self.prefix + suffix
        while True:
            page = await self._blobs.list_objects(prefix=prefix, limit=1000, cursor=cursor)
            for item in page.blobs:
                if not item.pathname.startswith(prefix):
                    raise RuntimeError("Storage returned an out-of-namespace object")
                yield item
            if not page.has_more:
                break
            if not page.cursor or page.cursor == cursor:
                raise RuntimeError("Storage pagination did not advance")
            cursor = page.cursor

    async def counts(self):
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        async for item in self.objects("context/"):
            scope = item.pathname[len(self.prefix):].split("/")[1]
            if scope in counts and item.pathname.endswith(".json"):
                counts[scope] += 1
        return counts

    async def clear(self):
        # Resolve this deployment's exact targets before deleting; never empty
        # the store or touch other projects/deployments sharing it.
        targets = [item.url async for item in self.objects()]
        for offset in range(0, len(targets), 100):
            await self._blobs.delete(targets[offset:offset + 100])

    async def aclose(self):
        await self._blobs.aclose()
        await self._http.aclose()
