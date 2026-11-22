"""Step 5 Task 4: read-only dataset integrity and distribution diagnostics.

Run from repository root:
  python engine/diagnose_dataset.py --src data/protocol_splits --filtered data/protocol_filtered --out data/diagnostics
Optional exact token lengths (requires installed tokenizer available locally):
  python engine/diagnose_dataset.py ... --tokenizer /path/to/local/tokenizer
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from engine.world import load_house
from engine.serialize import parse_output

SPLITS = ("train", "val", "test_a", "test_b")


def scan(path, house, *, expected_hash, expected_count, tokenizer=None, sample_limit=8):
    digest = hashlib.sha256()
    ids = set()
    counts = Counter()
    rooms = Counter()
    actions = Counter()
    samples = []
    token_lengths = []
    with path.open("rb") as stream:
        for lineno, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
                rid = record["id"]
                if not isinstance(rid, str) or rid in ids:
                    raise ValueError("invalid or duplicate record ID")
                ids.add(rid)
                if (
                    not isinstance(record["pid"], str)
                    or not isinstance(record["day"], int)
                    or not isinstance(record["minute"], int)
                ):
                    raise ValueError("invalid resident/time metadata")
                if not rid.startswith(record["pid"] + ":") or not (
                    0 <= record["minute"] < 1440
                ):
                    raise ValueError("inconsistent ID/time metadata")
                inp = yaml.safe_load(record["input"])
                if not isinstance(inp, dict):
                    raise ValueError("input YAML must be a mapping")
                if any(
                    key in inp
                    for key in (
                        "persona",
                        "preferences",
                        "label_do",
                        "next_room",
                        "decision",
                    )
                ):
                    raise ValueError("privileged field in input YAML")
                output = parse_output(house, record["output"])
                counts["examples"] += 1
                counts["action" if output["do"] else "idle"] += 1
                rooms[output["next"]] += 1
                actions.update(output["do"])
                if tokenizer is not None:
                    n = len(
                        tokenizer.encode(record["input"], add_special_tokens=False)
                    ) + len(
                        tokenizer.encode(record["output"], add_special_tokens=False)
                    )
                    token_lengths.append(n)
                    counts["combined_over_384"] += n > 384
                if len(samples) < sample_limit:
                    samples.append(record)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"{path}: SHA-256 mismatch")
    if counts["examples"] != expected_count:
        raise ValueError(
            f"{path}: count mismatch {counts['examples']} != {expected_count}"
        )
    report = {
        "examples": counts["examples"],
        "action": counts["action"],
        "idle": counts["idle"],
        "action_rate": round(counts["action"] / max(1, counts["examples"]), 6),
        "sha256": digest.hexdigest(),
        "next_room_counts": dict(sorted(rooms.items())),
        "action_counts": dict(sorted(actions.items())),
    }
    if tokenizer is not None and token_lengths:
        lengths = sorted(token_lengths)
        report["combined_tokens"] = {
            "median": lengths[len(lengths) // 2],
            "p95": lengths[min(len(lengths) - 1, int(0.95 * (len(lengths) - 1)))],
            "max": lengths[-1],
            "over_384": counts["combined_over_384"],
        }
    return report, ids, samples


def diagnose(src, filtered, out, house_path, tokenizer=None):
    house = load_house(str(house_path))
    original_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    filtered_manifest = json.loads(
        (filtered / "manifest.json").read_text(encoding="utf-8")
    )
    report = {
        "note": "Read-only checks. Token counts are omitted unless a tokenizer is supplied.",
        "splits": {},
        "filtered": {},
    }
    all_ids = {}
    samples = []
    for split in SPLITS:
        info = original_manifest["files"][split]
        result, ids, example_samples = scan(
            src / f"{split}.jsonl",
            house,
            expected_hash=info["sha256"],
            expected_count=info["examples"],
            tokenizer=tokenizer,
        )
        report["splits"][split] = result
        all_ids[split] = ids
        samples.extend({"split": split, **item} for item in example_samples)
        finfo = filtered_manifest["files"][split]
        filtered_result, filtered_ids, _ = scan(
            filtered / finfo["balanced_path"],
            house,
            expected_hash=finfo["balanced_sha256"],
            expected_count=finfo["balanced_examples"],
            tokenizer=tokenizer,
            sample_limit=0,
        )
        if not filtered_ids.issubset(ids):
            raise ValueError(f"{split}: filtered contains unknown IDs")
        if filtered_result["action"] != result["action"]:
            raise ValueError(f"{split}: action examples lost during filtering")
        if filtered_result["idle"] != finfo["balanced_idle"]:
            raise ValueError(f"{split}: idle count mismatch")
        report["filtered"][split] = filtered_result
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if not all_ids[a].isdisjoint(all_ids[b]):
                raise ValueError(f"overlapping record IDs: {a} and {b}")
    report["checks"] = {
        "hashes": "passed",
        "unique_ids": "passed",
        "split_id_overlap": "passed",
        "filtered_subset": "passed",
        "all_action_rows_retained": "passed",
        "yaml_parsing": "passed",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "samples_32.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples),
        encoding="utf-8",
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("data/protocol_splits"))
    p.add_argument("--filtered", type=Path, default=Path("data/protocol_filtered"))
    p.add_argument("--out", type=Path, default=Path("data/diagnostics"))
    p.add_argument("--house", type=Path, default=Path("engine/house_v1.yaml"))
    p.add_argument(
        "--tokenizer", help="Optional local Hugging Face tokenizer path; no downloads"
    )
    args = p.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = diagnose(args.src, args.filtered, args.out, args.house, tokenizer)
    print(
        json.dumps(
            {
                "checks": report["checks"],
                "original_counts": {
                    k: v["examples"] for k, v in report["splits"].items()
                },
                "filtered_counts": {
                    k: v["examples"] for k, v in report["filtered"].items()
                },
                "report": str(args.out / "report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
