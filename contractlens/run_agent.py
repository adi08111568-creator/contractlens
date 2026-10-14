#!/usr/bin/env python3
"""Run the extraction agent on contracts and cache the full results (fields + reasoning trace).

Setup (free):  get a key at https://aistudio.google.com  ->  export GEMINI_API_KEY=...
Usage:
    python run_agent.py --all                  # every contract in data/contracts/pdf
    python run_agent.py contract_01 contract_18
    python run_agent.py --all --sleep 8        # slower, for tight free-tier rate limits
Results go to data/cache/agent/<contract>.json and are skipped on re-run unless --force.
"""
import argparse
import json
import time
from pathlib import Path

from agent.agent import run_agent
from agent.doc import load_contract
from agent.llm import GeminiClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=None, help="Gemini model id (default: $GEMINI_MODEL or built-in)")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between contracts (rate limits)")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    pdfs = sorted((data / "contracts" / "pdf").glob("*.pdf"))
    todo = pdfs if a.all else [p for p in pdfs if p.stem in set(a.names)]
    if not todo:
        ap.error("give contract names or --all")
    out_dir = data / "cache" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient(model=a.model)
    print(f"model: {client.model} | contracts: {len(todo)}")

    for i, p in enumerate(todo, 1):
        out = out_dir / f"{p.stem}.json"
        if out.exists() and not a.force:
            print(f"[{i}/{len(todo)}] {p.stem}: cached, skipping")
            continue
        try:
            res = run_agent(load_contract(p), client, max_steps=a.max_steps)
        except Exception as e:  # keep going; one bad contract should not kill the batch
            print(f"[{i}/{len(todo)}] {p.stem}: FAILED {type(e).__name__}: {str(e)[:150]}")
            continue
        res["model"] = client.model
        out.write_text(json.dumps(res, indent=2, default=str))
        print(f"[{i}/{len(todo)}] {p.stem}: {res['steps']} steps, "
              f"{res['auto_fields']} auto / {res['review_fields']} review, "
              f"{res['validation_issues_caught']} issues caught, stop={res['stop_reason']}")
        time.sleep(a.sleep)
    print("done. Score with:  python evaluate.py --method both")


if __name__ == "__main__":
    main()
