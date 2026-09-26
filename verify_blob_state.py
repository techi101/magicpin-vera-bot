"""Small real-storage check; deletes only the random probe namespace it creates."""
import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv

from blob_state import PrivateBlobState, StorageConflict


async def main():
    load_dotenv(".env.storage-check")
    token = os.environ["BLOB_READ_WRITE_TOKEN"]
    namespace = "vera-storage-check-" + uuid4().hex
    one, two = PrivateBlobState(namespace, token), PrivateBlobState(namespace, token)
    try:
        assert await one.counts() == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        payload = {"slug": "probe", "summary": "Synthetic context for compression check. " * 200}
        status, _ = await one.put_context("category", "probe", {"version": 1, "payload": payload})
        assert status == "accepted"
        old, etag = await two.read_record("context:category:probe")
        assert old["version"] == 1
        print("Large-object ETag:", repr(etag), flush=True)
        status, _ = await two.put_context("category", "probe", {"version": 2, "payload": payload})
        assert status == "accepted"
        try:
            await one.conditional_put("context:category:probe", old, etag)
        except StorageConflict:
            pass
        else:
            raise AssertionError("ETag precondition was not enforced")
        assert (await one.get("context:category:probe"))["version"] == 2
        assert (await one.put_context("category", "probe", {"version": 2}))[0] == "duplicate"
        assert (await one.put_context("category", "probe", {"version": 1}))[0] == "stale"
        await asyncio.gather(*(one.put_context("customer", f"probe-{i}", {"version": 1}) for i in range(5)))
        assert (await two.counts())["customer"] == 5
        await one.set("probe-state", {"value": 1})
        await two.set("probe-state", {"value": 2})
        assert await one.get("probe-state") == {"value": 2}
        print("PASS private storage: cross-client reads, ETag conflict, version rules, exact counts, latest state")
    finally:
        await one.clear()
        assert sum((await two.counts()).values()) == 0
        await one.aclose()
        await two.aclose()
        print("PASS probe namespace cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
