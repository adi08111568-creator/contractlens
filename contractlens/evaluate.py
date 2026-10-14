#!/usr/bin/env python3
"""Score extraction results against the ground-truth answer key.

Usage:
    python evaluate.py --method rules                 # rules baseline on PDF text
    python evaluate.py --method agent                 # score cached agent runs (data/cache/agent/*.json)
    python evaluate.py --method both                  # side-by-side
Options: --input pdf|txt   --data data   --json out.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from agent.tools import FIELD_NAMES


def _n(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def equal(field, pred, truth):
    if pred is None:
        return False
    if isinstance(truth, bool):
        return isinstance(pred, bool) and pred == truth
    if isinstance(truth, int):
        try:
            return int(pred) == truth
        except (TypeError, ValueError):
            return False
    return _n(pred) == _n(truth)


def score(preds: dict, key: dict):
    """preds: {contract_id: {field: value}}; key: ground_truth.json. Contracts missing from preds are skipped."""
    per_field, per_layout = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    per_contract, total = {}, [0, 0]
    for cid, p in preds.items():
        rec = key[cid]
        ok = 0
        for f in FIELD_NAMES:
            good = equal(f, p.get(f), rec["truth"][f])
            ok += good
            per_field[f][0] += good
            per_field[f][1] += 1
            per_layout[rec["layout"]][0] += good
            per_layout[rec["layout"]][1] += 1
        per_contract[cid] = ok / len(FIELD_NAMES)
        total[0] += ok
        total[1] += len(FIELD_NAMES)
    pct = lambda a: round(100 * a[0] / a[1], 1) if a[1] else None
    return {"overall": pct(total), "n_contracts": len(preds), "n_fields": total[1],
            "by_field": {f: pct(v) for f, v in per_field.items()},
            "by_layout": {k: pct(v) for k, v in sorted(per_layout.items())},
            "per_contract": per_contract}


def rules_preds(data: Path, input_kind: str):
    import baseline_rules
    from agent.doc import load_contract
    ext = "pdf" if input_kind == "pdf" else "txt"
    return {p.stem: baseline_rules.extract(load_contract(p).text)
            for p in sorted((data / "contracts" / ext).glob(f"*.{ext}"))}


def agent_preds(data: Path):
    preds = {}
    for p in sorted((data / "cache" / "agent").glob("*.json")):
        r = json.loads(p.read_text())
        preds[r["contract"]] = {f: v["value"] for f, v in r["fields"].items()}
    return preds


def agent_routing(data: Path, key: dict):
    """How good is the agent's own confidence routing? auto-accepted accuracy vs review rate."""
    auto_ok = auto_n = review_n = review_would_be_ok = total = 0
    for p in sorted((data / "cache" / "agent").glob("*.json")):
        r = json.loads(p.read_text())
        for f, v in r["fields"].items():
            total += 1
            good = equal(f, v["value"], key[r["contract"]]["truth"][f])
            if v["status"] == "auto":
                auto_n += 1
                auto_ok += good
            else:
                review_n += 1
                review_would_be_ok += good
    if not total:
        return None
    return {"fields_total": total,
            "auto_accepted_pct": round(100 * auto_n / total, 1),
            "auto_accuracy_pct": round(100 * auto_ok / auto_n, 1) if auto_n else None,
            "sent_to_review_pct": round(100 * review_n / total, 1),
            "review_fields_that_were_actually_correct": review_would_be_ok}


def print_report(name, s):
    print(f"\n=== {name}: {s['overall']}% field accuracy "
          f"({s['n_contracts']} contracts, {s['n_fields']} fields) ===")
    print("  by layout:", "  ".join(f"{k.split('_')[0]}={v}%" for k, v in s["by_layout"].items()))
    for f, v in sorted(s["by_field"].items(), key=lambda x: x[1]):
        print(f"  {f:<28}{v:>6}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rules", "agent", "both"], default="both")
    ap.add_argument("--input", choices=["pdf", "txt"], default="pdf")
    ap.add_argument("--data", default="data")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    key = json.loads((data / "ground_truth.json").read_text())
    report = {}
    if a.method in ("rules", "both"):
        report["rules"] = score(rules_preds(data, a.input), key)
        print_report("Rules baseline", report["rules"])
    if a.method in ("agent", "both"):
        preds = agent_preds(data)
        if preds:
            report["agent"] = score(preds, key)
            report["agent_routing"] = agent_routing(data, key)
            print_report("Agent", report["agent"])
            print("  routing:", report["agent_routing"])
        else:
            print("\nNo cached agent runs in data/cache/agent/. Run:  python run_agent.py --all")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
