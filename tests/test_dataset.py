"""Tests for the Step 5 pilot dataset generator."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from build_dataset import build
from serialize import parse_output
from world import load_house


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pilot_is_reproducible_and_resident_disjoint(tmp_path):
    house_path = ROOT / "engine" / "house_v1.yaml"
    a, b = tmp_path / "a", tmp_path / "b"
    ma = build(house_path=house_path, out=a)
    mb = build(house_path=house_path, out=b)
    assert ma == mb
    assert all((a / name).read_bytes() == (b / name).read_bytes()
               for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"))
    house = load_house(str(house_path))
    seen = set()
    for split in ("train", "val", "test"):
        entries = read_jsonl(a / f"{split}.jsonl")
        assert len(entries) == 2 * (1440 // house.tick)
        ids = {e["pid"] for e in entries}
        assert not ids & seen
        seen |= ids
        for e in entries:
            assert set(e) == {"pid", "day", "minute", "input", "output"}
            assert "setpoint_home" not in e["input"]
            assert "next_act" not in e["input"]
            parse_output(house, e["output"])
    assert len(seen) == 3


def test_invalid_split_rejected(tmp_path):
    with pytest.raises(ValueError):
        build(house_path=ROOT / "engine" / "house_v1.yaml", out=tmp_path,
              residents=3, n_val=2, n_test=1)


def test_protocol_temporal_and_resident_isolation(tmp_path):
    from build_dataset import build_protocol
    manifest = build_protocol(house_path=ROOT / "engine" / "house_v1.yaml",
                              out=tmp_path, residents=4, days=3, train_days=2,
                              n_val=1, n_test=1, seed=0)
    groups = manifest["resident_splits"]
    assert [len(groups[k]) for k in ("train", "val", "test")] == [2, 1, 1]
    expected = {"train": 576, "test_a": 288, "val": 432, "test_b": 432}
    assert {k: v["examples"] for k, v in manifest["files"].items()} == expected
    ids_by_split = {}
    all_ids = set()
    for split in expected:
        rows = read_jsonl(tmp_path / f"{split}.jsonl")
        ids_by_split[split] = {r["pid"] for r in rows}
        for row in rows:
            assert row["id"] == f"{row['pid']}:d{row['day']:03d}:m{row['minute']:04d}"
            assert row["id"] not in all_ids
            all_ids.add(row["id"])
            if split == "train":
                assert row["day"] < 2
            elif split == "test_a":
                assert row["day"] >= 2
    assert ids_by_split["train"] == ids_by_split["test_a"]
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test_b"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test_b"])


def test_protocol_determinism(tmp_path):
    from build_dataset import build_protocol
    kw = dict(house_path=ROOT / "engine" / "house_v1.yaml", residents=4,
              days=3, train_days=2, n_val=1, n_test=1, seed=0)
    a = build_protocol(out=tmp_path / "a", **kw)
    b = build_protocol(out=tmp_path / "b", **kw)
    assert a == b
    for split in ("train", "val", "test_a", "test_b"):
        assert (tmp_path / "a" / f"{split}.jsonl").read_bytes() == (tmp_path / "b" / f"{split}.jsonl").read_bytes()
