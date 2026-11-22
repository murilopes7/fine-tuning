import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from scripts.dataset.filter_dataset import canonical, filter_dataset


def make_source(root):
    root.mkdir()
    files = {}
    for split in ("train", "val", "test_a", "test_b"):
        records = [
            {
                "id": f"{split}-{i}",
                "pid": split,
                "day": i // 144,
                "minute": i % 144 * 10,
                "input": "state",
                "output": "next: living\ndo: [light.living=on]"
                if i % 3 == 0
                else "next: living\ndo: []",
            }
            for i in range(100)
        ]
        raw = b"".join(canonical(r) for r in records)
        (root / f"{split}.jsonl").write_bytes(raw)
        files[split] = {
            "examples": len(records),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps({"files": files}))


def test_filter_counts_and_reproducibility(tmp_path):
    src = tmp_path / "source"
    make_source(src)
    a = filter_dataset(src, tmp_path / "a")
    b = filter_dataset(src, tmp_path / "b")
    assert a == b
    for split in ("train", "val", "test_a", "test_b"):
        info = a["files"][split]
        assert info["balanced_actions"] == 34
        assert info["balanced_idle"] == round(66 * 0.05)
        assert (tmp_path / "a" / f"{split}_balanced.jsonl").read_bytes() == (
            tmp_path / "b" / f"{split}_balanced.jsonl"
        ).read_bytes()
        assert (src / f"{split}.jsonl").exists()


def test_refuses_source_hash_mismatch(tmp_path):
    src = tmp_path / "source"
    make_source(src)
    path = src / "train.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b" \n", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        filter_dataset(src, tmp_path / "out")


def test_refuses_same_directory(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        filter_dataset(tmp_path, tmp_path)
