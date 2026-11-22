"""Step 5 pilot: deterministic, resident-disjoint synthetic dataset.

Run from repository root: python engine/build_dataset.py --out data/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from engine.resident import sample_persona, simulate, split_residents
from engine.teacher import oracle
from engine.world import load_house
from engine.serialize import serialize_examples, parse_output


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def build(
    *,
    house_path: Path,
    out: Path,
    residents: int = 3,
    days: int = 2,
    seed: int = 7,
    n_val: int = 1,
    n_test: int = 1,
    noise_c: float = 0.3,
    p_missing: float = 0.0,
    history_window: int = 12,
) -> dict:
    if (
        residents < 3
        or days < 1
        or n_val < 1
        or n_test < 1
        or residents <= n_val + n_test
    ):
        raise ValueError(
            "need at least 3 residents, 1 day, and nonempty train/val/test groups"
        )
    if not 0 <= p_missing <= 1 or noise_c < 0:
        raise ValueError("invalid observation noise settings")
    house = load_house(str(house_path))
    # Single shared generator produces diverse but reproducible personas.
    persona_rng = random.Random(seed)
    personas = [sample_persona(persona_rng, i) for i in range(residents)]
    splits = split_residents(personas, seed=seed, n_val=n_val, n_test=n_test)
    id_to_split = {pid: split for split, ids in splits.items() for pid in ids}
    records = {split: [] for split in ("train", "val", "test")}
    for i, persona in enumerate(personas):
        rows = simulate(house, persona, days, seed=seed, teacher=oracle, follow=False)
        examples = serialize_examples(
            house,
            rows,
            seed=seed + i,
            noise_c=noise_c,
            p_missing=p_missing,
            history_window=history_window,
        )
        assert len(rows) == len(examples)
        for row, example in zip(rows, examples):
            parsed = parse_output(house, example["output"])
            assert parsed["next"] == row["next_room"]
            assert parsed["do"] == sorted(row["label_do"])
            records[id_to_split[persona.pid]].append(
                {
                    "pid": persona.pid,
                    "day": row["day"],
                    "minute": row["minute"],
                    "input": example["input"],
                    "output": example["output"],
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    file_info = {}
    for split, entries in records.items():
        data = b"".join(canonical_bytes(e) for e in entries)
        path = out / f"{split}.jsonl"
        path.write_bytes(data)
        file_info[split] = {
            "examples": len(entries),
            "sha256": hashlib.sha256(data).hexdigest(),
            "nonempty_actions": sum(
                bool(parse_output(house, e["output"])["do"]) for e in entries
            ),
            "next_room_counts": dict(
                sorted(
                    Counter(
                        parse_output(house, e["output"])["next"] for e in entries
                    ).items()
                )
            ),
        }
    manifest = {
        "generator": "step5-pilot-v1",
        "seed": seed,
        "residents": residents,
        "days_per_resident": days,
        "ticks_per_day": 1440 // house.tick,
        "noise_c": noise_c,
        "p_missing": p_missing,
        "history_window": history_window,
        "teacher": "oracle",
        "follow": False,
        "house_sha256": hashlib.sha256(house_path.read_bytes()).hexdigest(),
        "resident_splits": splits,
        "files": file_info,
        "notes": "Pilot only; no temporal holdout (Test A) or sensor-stress test (Test C) yet. Persona preferences are never written to model inputs.",
    }
    (out / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--house", type=Path, default=Path(__file__).with_name("house_v1.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    parser.add_argument("--residents", type=int, default=3)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-residents", type=int, default=1)
    parser.add_argument("--test-residents", type=int, default=1)
    parser.add_argument("--noise-c", type=float, default=0.3)
    parser.add_argument("--p-missing", type=float, default=0.0)
    parser.add_argument("--history-window", type=int, default=12)
    args = parser.parse_args()
    manifest = build(
        house_path=args.house,
        out=args.out,
        residents=args.residents,
        days=args.days,
        seed=args.seed,
        n_val=args.val_residents,
        n_test=args.test_residents,
        noise_c=args.noise_c,
        p_missing=args.p_missing,
        history_window=args.history_window,
    )
    print(
        json.dumps(
            {
                "resident_splits": manifest["resident_splits"],
                "files": manifest["files"],
            },
            indent=2,
        )
    )


# Protocol profile is intentionally separate from the Step 5 pilot.
# The protocol remains draft: validation and Test B use days 0..79
# provisionally, until the advisor approves those day ranges.
def build_protocol(
    *,
    house_path: Path,
    out: Path,
    seed: int = 0,
    residents: int = 30,
    days: int = 80,
    train_days: int = 60,
    n_val: int = 4,
    n_test: int = 6,
    noise_c: float = 0.3,
    p_missing: float = 0.0,
    history_window: int = 12,
) -> dict:
    """Create natural-distribution splits; filtering and Test C are later tasks.

    Training residents: days [0, train_days) -> train;
    training residents: days [train_days, days) -> test_a;
    validation residents: all days -> val;
    held-out residents: all days -> test_b.
    """
    if not (
        residents > n_val + n_test
        and n_val > 0
        and n_test > 0
        and 0 < train_days < days
    ):
        raise ValueError("invalid resident counts or temporal split")
    if not 0 <= p_missing <= 1 or noise_c < 0:
        raise ValueError("invalid observation settings")
    house = load_house(str(house_path))
    persona_rng = random.Random(seed)
    personas = [sample_persona(persona_rng, i) for i in range(residents)]
    splits = split_residents(personas, seed=seed, n_val=n_val, n_test=n_test)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    id_to_group = {pid: group for group, ids in splits.items() for pid in ids}
    names = ("train", "val", "test_a", "test_b")
    out.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in names}
    action_counts = {name: 0 for name in names}
    room_counts = {name: Counter() for name in names}
    digests = {name: hashlib.sha256() for name in names}
    paths = {name: out / f"{name}.jsonl" for name in names}
    # Stream output to avoid holding ~345,600 serialized examples in memory.
    from contextlib import ExitStack

    with ExitStack() as stack:
        handles = {name: stack.enter_context(paths[name].open("wb")) for name in names}
        for i, persona in enumerate(personas):
            rows = simulate(
                house, persona, days, seed=seed, teacher=oracle, follow=False
            )
            examples = serialize_examples(
                house,
                rows,
                seed=seed + i,
                noise_c=noise_c,
                p_missing=p_missing,
                history_window=history_window,
            )
            if len(rows) != days * (1440 // house.tick) or len(rows) != len(examples):
                raise AssertionError("unexpected simulation length")
            for row, example in zip(rows, examples):
                group = id_to_group[persona.pid]
                split = (
                    ("train" if row["day"] < train_days else "test_a")
                    if group == "train"
                    else ("val" if group == "val" else "test_b")
                )
                parsed = parse_output(house, example["output"])
                if parsed["next"] != row["next_room"]:
                    raise AssertionError("next-room label mismatch")
                record = {
                    "id": f"{persona.pid}:d{row['day']:03d}:m{row['minute']:04d}",
                    "pid": persona.pid,
                    "day": row["day"],
                    "minute": row["minute"],
                    "input": example["input"],
                    "output": example["output"],
                }
                raw = canonical_bytes(record)
                handles[split].write(raw)
                digests[split].update(raw)
                counts[split] += 1
                action_counts[split] += bool(parsed["do"])
                room_counts[split][parsed["next"]] += 1
    per_day = 1440 // house.tick
    expected = {
        "train": len(splits["train"]) * train_days * per_day,
        "test_a": len(splits["train"]) * (days - train_days) * per_day,
        "val": len(splits["val"]) * days * per_day,
        "test_b": len(splits["test"]) * days * per_day,
    }
    if counts != expected:
        raise AssertionError(f"split count mismatch: {counts} != {expected}")
    manifest = {
        "generator": "step5-protocol-splits-v1",
        "protocol_status": "DRAFT; not frozen",
        "seed": seed,
        "residents": residents,
        "days_per_resident": days,
        "train_days": [0, train_days - 1],
        "test_a_days": [train_days, days - 1],
        "val_days": [0, days - 1],
        "test_b_days": [0, days - 1],
        "ticks_per_day": per_day,
        "resident_splits": splits,
        "noise_c": noise_c,
        "p_missing": p_missing,
        "history_window": history_window,
        "teacher": "oracle",
        "follow": False,
        "house_sha256": hashlib.sha256(house_path.read_bytes()).hexdigest(),
        "files": {
            name: {
                "examples": counts[name],
                "sha256": digests[name].hexdigest(),
                "nonempty_actions": action_counts[name],
                "next_room_counts": dict(sorted(room_counts[name].items())),
            }
            for name in names
        },
        "notes": [
            "Natural distribution; action/idle filtering and deduplication not yet applied.",
            "Test C is not generated yet.",
            "Validation and Test B day ranges are provisional pending protocol approval.",
            "Resident history is serialized across the entire simulation, including preceding days; no training rows from Test A are included in train.",
        ],
    }
    (out / "manifest.json").write_bytes(canonical_bytes(manifest))
    return manifest


def protocol_main():
    parser = argparse.ArgumentParser(
        description="Step 5 protocol-aligned split generation"
    )
    parser.add_argument(
        "--house", type=Path, default=Path(__file__).with_name("house_v1.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("data/protocol_splits"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residents", type=int, default=30)
    parser.add_argument("--days", type=int, default=80)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-residents", type=int, default=4)
    parser.add_argument("--test-residents", type=int, default=6)
    args = parser.parse_args()
    manifest = build_protocol(
        house_path=args.house,
        out=args.out,
        seed=args.seed,
        residents=args.residents,
        days=args.days,
        train_days=args.train_days,
        n_val=args.val_residents,
        n_test=args.test_residents,
    )
    print(
        json.dumps(
            {
                "resident_splits": manifest["resident_splits"],
                "files": manifest["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "protocol":
        sys.argv.pop(1)
        protocol_main()
    else:
        main()
