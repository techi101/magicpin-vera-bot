"""Run the supplied simulator's four non-LLM scenarios without fabricating scores.

Usage: python run_simulator_contract.py http://127.0.0.1:8080
This resets ONLY the explicitly targeted challenge deployment before and after.
"""
import json
import sys
from urllib import request

import judge_simulator as supplied


class NoScoringProvider(supplied.LLMProvider):
    def name(self):
        return "Protocol checks only; LLM scoring NOT run"

    def complete(self, prompt, system=None):
        raise RuntimeError("This runner does not simulate or fabricate an LLM judge score")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_simulator_contract.py BOT_URL")
    supplied.BOT_URL = sys.argv[1].rstrip("/")

    def reset():
        req = request.Request(supplied.BOT_URL + "/v1/teardown", data=b"{}", headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=30) as response:
            assert json.load(response)["wiped"]

    reset()
    try:
        # Original 'all' runs warmup, auto-reply, intent and hostile checks only.
        success = supplied.JudgeSimulator(NoScoringProvider()).run("all")
    finally:
        reset()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
