import asyncio
import fnmatch
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from redis_state import PUT_CONTEXT_LUA, RedisState


class FakeRedis:
    """In-memory Redis subset. EVAL runs the Lua scripts' semantics atomically."""
    def __init__(self):
        self.data, self.sets, self.calls = {}, {}, []

    def run(self, args):
        self.calls.append(args[0])
        op = args[0]
        if op == "GET":
            return self.data.get(args[1])
        if op == "SET":
            self.data[args[1]] = args[2]; return "OK"
        if op == "DEL":
            n = 0
            for key in args[1:]:
                n += key in self.data or key in self.sets
                self.data.pop(key, None); self.sets.pop(key, None)
            return n
        if op == "SCAN":
            pattern = args[3]
            return ["0", [k for k in list(self.data) + list(self.sets) if fnmatch.fnmatchcase(k, pattern)]]
        if op == "EVAL":
            script, nkeys = args[1], int(args[2])
            keys, argv = args[3:3 + nkeys], args[3 + nkeys:]
            if script == PUT_CONTEXT_LUA:
                current = self.data.get(keys[0])
                if current is not None:
                    old, new = json.loads(current)["version"], int(argv[1])
                    if old >= new:
                        return ["duplicate" if old == new else "stale", current]
                self.data[keys[0]] = argv[0]
                self.sets.setdefault(keys[1], set()).add(argv[2])
                return ["accepted", argv[0]]
            if "SCARD" in script:
                return [len(self.sets.get(k, ())) for k in keys]
        raise AssertionError(f"unexpected command {args}")


def worker(backend, namespace="vera-test"):
    state = RedisState(namespace, "https://example.upstash.io", "token")

    async def command(*args):
        await asyncio.sleep(0)
        return backend.run([str(a) for a in args])
    state.command = command
    return state


class RedisStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_versions_never_regress_across_workers(self):
        backend = FakeRedis()
        low, high = worker(backend), worker(backend)
        await asyncio.gather(low.put_context("customer", "c1", {"version": 1}),
                             high.put_context("customer", "c1", {"version": 2}))
        self.assertEqual((await low.get("context:customer:c1"))["version"], 2)
        self.assertEqual((await low.put_context("customer", "c1", {"version": 1}))[0], "stale")
        self.assertEqual((await low.put_context("customer", "c1", {"version": 2}))[0], "duplicate")

    async def test_255_concurrent_contexts_counted_exactly_without_listing(self):
        backend = FakeRedis()
        records = [("category", f"k{i}") for i in range(5)] + [("merchant", f"m{i}") for i in range(50)] + \
                  [("customer", f"c{i}") for i in range(200)]
        await asyncio.gather(*(worker(backend).put_context(s, i, {"version": 1}) for s, i in records))
        await asyncio.gather(*(worker(backend).put_context(s, i, {"version": 1}) for s, i in records[:20]))
        counts = await worker(backend).counts()
        self.assertEqual(counts, {"category": 5, "merchant": 50, "customer": 200, "trigger": 0})
        self.assertEqual(backend.calls.count("EVAL"), 255 + 20 + 1)

    async def test_state_roundtrip_and_namespaced_clear(self):
        backend = FakeRedis()
        ours, other = worker(backend, "vera-a"), worker(backend, "vera-b")
        await ours.set("conv:1", {"turns": [1]})
        await ours.put_context("trigger", "t1", {"version": 1})
        await other.set("conv:1", {"keep": True})
        self.assertEqual(await ours.get("conv:1"), {"turns": [1]})
        await ours.clear()
        self.assertIsNone(await ours.get("conv:1"))
        self.assertEqual((await ours.counts())["trigger"], 0)
        self.assertEqual(await other.get("conv:1"), {"keep": True})

    async def test_transient_http_failures_retry_then_succeed(self):
        state = RedisState("vera-test", "https://example.upstash.io", "token")
        ok = httpx.Response(200, json={"result": "OK"}, request=httpx.Request("POST", "https://x"))
        busy = httpx.Response(503, request=httpx.Request("POST", "https://x"))
        state._http.post = AsyncMock(side_effect=[httpx.ReadTimeout("slow"), busy, ok])
        with patch("redis_state.asyncio.sleep", AsyncMock()):
            self.assertEqual(await state.command("SET", "k", "v"), "OK")
        self.assertEqual(state._http.post.await_count, 3)
        await state.aclose()

    async def test_auth_errors_and_redis_errors_are_not_retried(self):
        state = RedisState("vera-test", "https://example.upstash.io", "token")
        denied = httpx.Response(401, request=httpx.Request("POST", "https://x"))
        state._http.post = AsyncMock(return_value=denied)
        with self.assertRaises(httpx.HTTPStatusError):
            await state.command("GET", "k")
        self.assertEqual(state._http.post.await_count, 1)
        state._http.post = AsyncMock(return_value=httpx.Response(
            200, json={"error": "ERR bad"}, request=httpx.Request("POST", "https://x")))
        with self.assertRaises(RuntimeError):
            await state.command("GET", "k")
        self.assertEqual(state._http.post.await_count, 1)
        await state.aclose()

    def test_unsafe_namespace_and_scope_rejected(self):
        with self.assertRaises(ValueError):
            RedisState("bad:ns*", "https://x", "t")
        with self.assertRaises(ValueError):
            RedisState("vera", "", "t")
        with self.assertRaises(ValueError):
            RedisState("vera", "https://x", "t").context_key("admin", "x")


if __name__ == "__main__":
    unittest.main()
