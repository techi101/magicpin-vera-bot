"""Create a reproducible, allowlisted submission ZIP; exclude credentials and environments."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).parent


def main():
    subprocess.run([sys.executable, "build_submission.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "-q", "test_bot.py", "test_blob_state.py", "test_redis_state.py"], cwd=ROOT, check=True)
    report_path = ROOT / "artifacts/live-audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] and report["version"] == "2.2.1"
    assert report["bot_sha256"] == hashlib.sha256((ROOT / "bot.py").read_bytes()).hexdigest(), "Live audit is stale; rerun it after code changes."
    for name in ["bot.py", "blob_state.py", "redis_state.py"]:
        assert report["source_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), f"Live audit is stale for {name}."
    files = [ROOT / name for name in [
        "bot.py", "blob_state.py", "redis_state.py", "submission.jsonl", "README.md", "requirements.txt", ".python-version",
        "SUBMISSION.md", "vercel.json", ".vercelignore", ".gitignore", "api/index.py",
        "build_submission.py", "test_bot.py", "test_blob_state.py", "test_redis_state.py", "verify_blob_state.py", "test_dashboard.cjs",
        "verify_deployment.py", "verify_adaptive.py", "submission_audit.py",
        "run_simulator_contract.py", "package_submission.py", "judge_simulator.py",
        "challenge-brief.md", "challenge-testing-brief.md", "artifacts/live-audit.json",
    ]]
    files.extend(sorted((ROOT / "dataset").rglob("*.json")))
    files.append(ROOT / "dataset/generate_dataset.py")
    entries = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert not any(p in {".vercel", ".venv", "__pycache__"} or p.startswith(".env") for p in path.relative_to(ROOT).parts)
        data = path.read_bytes()
        entries[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"version": "2.2.1", "created_at": datetime.now(timezone.utc).isoformat(), "base_url": report["base_url"], "portal_submitted": False, "files": entries}
    archive = ROOT / "magicpin-submission-v2.2.1.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name, info in entries.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == info["sha256"]
    print(json.dumps({"archive": str(archive), "files": len(entries) + 1, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "portal_submitted": False}, indent=2))


if __name__ == "__main__":
    main()
