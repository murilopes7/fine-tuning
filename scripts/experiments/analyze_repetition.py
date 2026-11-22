"""Read-only analysis of existing Step 5 JSONL and supplementary diagnostic report.

Run: python engine/analyze_repetition.py --src data/protocol_splits/train.jsonl --out data/diagnostics/repetition.json
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from engine.world import load_house
from engine.serialize import parse_output


def analyze(path, house):
    previous = {}
    totals = Counter()
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        pid = row["pid"]
        day, minute = row["day"], row["minute"]
        parsed = parse_output(house, row["output"])
        actions = set(parsed["do"])
        totals["examples"] += 1
        if actions:
            totals["action_rows"] += 1
            totals["command_instances"] += len(actions)
        prev = previous.get(pid)
        if prev is not None:
            pday, pminute, pnext, pactions = prev
            adjacent = day * 1440 + minute - (pday * 1440 + pminute) == house.tick
            if adjacent and actions:
                if (parsed["next"], actions) == (pnext, pactions):
                    totals["repeated_full_outputs"] += 1
                if actions == pactions:
                    totals["repeated_action_sets"] += 1
                intersection = actions & pactions
                totals["repeated_command_instances"] += len(intersection)
                if intersection:
                    totals["action_rows_sharing_any_previous_command"] += 1
            elif not adjacent:
                totals["nonadjacent_comparisons_skipped"] += 1
        previous[pid] = (day, minute, parsed["next"], actions)
    n = totals["action_rows"]
    c = totals["command_instances"]
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": dict(totals),
        "rates": {
            "identical_full_output_per_action_row": round(
                totals["repeated_full_outputs"] / n, 6
            )
            if n
            else None,
            "identical_action_set_per_action_row": round(
                totals["repeated_action_sets"] / n, 6
            )
            if n
            else None,
            "action_rows_sharing_command_per_action_row": round(
                totals["action_rows_sharing_any_previous_command"] / n, 6
            )
            if n
            else None,
            "repeated_command_instances_per_command_instance": round(
                totals["repeated_command_instances"] / c, 6
            )
            if c
            else None,
        },
        "prior_exploratory_result": {
            "repeated_full_outputs": 93767,
            "action_rows": 114558,
            "rate_percent_rounded": 81.9,
            "caveat": "Earlier script compared previous stored output for each resident, without checking timestamp adjacency. This new analysis checks adjacency and parses YAML actions.",
        },
        "interpretation": "Consecutive identical labels measure repetition, not necessarily unsafe or redundant physical actuation.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("data/protocol_splits/train.jsonl"))
    p.add_argument(
        "--house", type=Path, default=Path(__file__).with_name("house_v1.yaml")
    )
    p.add_argument("--out", type=Path, default=Path("data/diagnostics/repetition.json"))
    a = p.parse_args()
    report = analyze(a.src, load_house(str(a.house)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
