import asyncio
import copy
import json
import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from blob_state import PrivateBlobState, StorageConflict


class MemoryCAS(PrivateBlobState):
    """Independent workers sharing only the backend, not Python locks."""
    def __init__(self, backend):
        self.backend = backend

    async def read_record(self, key):
        record, version = self.backend.get(key, (None, None))
        await asyncio.sleep(0)
        return copy.deepcopy(record), version

    async def conditional_put(self, key, value, etag):
        current = self.backend.get(key, (None, None))[1]
        if current != etag:
            raise StorageConflict()
        self.backend[key] = (copy.deepcopy(value), (current or 0) + 1)


class BlobStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_competing_versions_cannot_regress(self):
        backend = {}
        low, high = MemoryCAS(backend), MemoryCAS(backend)
        await asyncio.gather(low.put_context("customer", "new-id", {"version": 1}),
                             high.put_context("customer", "new-id", {"version": 2}))
        self.assertEqual((await high.get("context:customer:new-id"))["version"], 2)
        status, stored = await low.put_context("customer", "new-id", {"version": 1})
        self.assertEqual((status, stored["version"]), ("stale", 2))

    async def test_duplicate_version_is_noop_across_workers(self):
        backend = {}
        workers = [MemoryCAS(backend) for _ in range(5)]
        outcomes = await asyncio.gather(*(w.put_context("category", "new-id", {"version": 1, "payload": i})
                                         for i, w in enumerate(workers)))
        self.assertEqual(sum(status == "accepted" for status, _ in outcomes), 1)
        self.assertEqual(sum(status == "duplicate" for status, _ in outcomes), 4)
        self.assertEqual(len({stored["payload"] for _, stored in outcomes}), 1)

    async def test_concurrent_distinct_contexts_have_no_shared_index(self):
        backend = {}
        await asyncio.gather(*(MemoryCAS(backend).put_context("customer", f"id-{i}", {"version": 1})
                               for i in range(255)))
        self.assertEqual(len(backend), 255)

    async def test_transient_upstream_failures_retry_idempotently(self):
        backend = {}
        worker = MemoryCAS(backend)
        real_put = worker.conditional_put
        calls = []

        async def flaky_put(key, value, etag):
            calls.append(etag)
            if len(calls) == 1:
                raise httpx.HTTPStatusError("503", request=httpx.Request("PUT", "https://x"),
                                            response=httpx.Response(503))
            if len(calls) == 2:
                await real_put(key, value, etag)  # write lands, response lost
                raise httpx.ReadTimeout("lost response")
            await real_put(key, value, etag)
        worker.conditional_put = flaky_put
        with unittest.mock.patch("blob_state.asyncio.sleep", AsyncMock()):
            status, stored = await worker.put_context("customer", "flaky", {"version": 1})
        self.assertEqual((status, stored["version"], len(calls)), ("duplicate", 1, 2))

    async def test_permanent_errors_are_not_retried(self):
        worker = MemoryCAS({})
        worker.conditional_put = AsyncMock(side_effect=httpx.HTTPStatusError(
            "403", request=httpx.Request("PUT", "https://x"), response=httpx.Response(403)))
        with self.assertRaises(httpx.HTTPStatusError):
            await worker.put_context("customer", "denied", {"version": 1})
        self.assertEqual(worker.conditional_put.await_count, 1)

    async def test_consistent_read_bypasses_cache(self):
        store = object.__new__(PrivateBlobState)
        store.prefix, store._token, store._origin = "test/", "test-token", "https://test.invalid"
        requests = []

        def respond(req):
            requests.append(req)
            return httpx.Response(200, json={"version": 3}, headers={"etag": '"e3"'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store._http = client
            self.assertEqual(await store.read_record("context:category:new"), ({"version": 3}, '"e3"'))
        self.assertEqual(requests[0].url.params["cache"], "0")
        self.assertEqual(requests[0].headers["accept-encoding"], "identity")

    async def test_weak_etag_cannot_be_used_for_conditional_update(self):
        store = object.__new__(PrivateBlobState)
        store.prefix, store._token, store._origin = "test/", "test-token", "https://test.invalid"
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"version": 1}, headers={"etag": 'W/"weak"'}))) as client:
            store._http = client
            with self.assertRaisesRegex(RuntimeError, "strong"):
                await store.read_record("context:category:new")

    async def test_conditional_write_headers_and_conflicts(self):
        store = object.__new__(PrivateBlobState)
        store.prefix, store._token = "test/", "test-token"
        requests = []

        def respond(req):
            requests.append(req)
            return httpx.Response(412, json={"error": {"code": "precondition_failed"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store._http = client
            with self.assertRaises(StorageConflict):
                await store.conditional_put("context:category:new", {"version": 2}, '"old"')
        self.assertEqual(requests[0].headers["x-if-match"], '"old"')
        self.assertEqual(requests[0].headers["x-allow-overwrite"], "1")
        self.assertEqual(requests[0].headers["x-vercel-blob-access"], "private")

    async def test_paginated_counts_and_namespace_limited_teardown(self):
        store = object.__new__(PrivateBlobState)
        store.prefix = "test/"
        pages = [SimpleNamespace(blobs=[SimpleNamespace(pathname="test/context/customer/1.json", url="u1")], has_more=True, cursor="next"),
                 SimpleNamespace(blobs=[SimpleNamespace(pathname="test/context/customer/2.json", url="u2")], has_more=False, cursor=None)]
        store._blobs = SimpleNamespace(list_objects=AsyncMock(side_effect=pages + pages), delete=AsyncMock())
        self.assertEqual(await store.counts(), {"category": 0, "merchant": 0, "customer": 2, "trigger": 0})
        await store.clear()
        store._blobs.delete.assert_awaited_once_with(["u1", "u2"])
        self.assertEqual(store._blobs.list_objects.call_args.kwargs["prefix"], "test/")

    async def test_teardown_rejects_out_of_namespace_targets(self):
        store = object.__new__(PrivateBlobState)
        store.prefix = "test/"
        page = SimpleNamespace(blobs=[SimpleNamespace(pathname="someone-else/data.json", url="private")], has_more=False)
        store._blobs = SimpleNamespace(list_objects=AsyncMock(return_value=page), delete=AsyncMock())
        with self.assertRaises(RuntimeError):
            await store.clear()
        store._blobs.delete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
