import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from scripts.dataset.diagnose_dataset import scan
from engine.world import load_house

HOUSE = load_house(str(ROOT / "engine" / "house_v1.yaml"))


def make_file(tmp_path, records):
    path = tmp_path / "test.jsonl"
    raw = b"".join((json.dumps(x) + "\n").encode() for x in records)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_valid_example(tmp_path):
    record = {
        "id": "r00:d000:m0000",
        "pid": "r00",
        "day": 0,
        "minute": 0,
        "input": "time: Mon 00:00\n",
        "output": "next: kitchen\ndo: []\n",
    }
    path, sha = make_file(tmp_path, [record])
    report, ids, samples = scan(path, HOUSE, expected_hash=sha, expected_count=1)
    assert report["idle"] == 1 and report["action"] == 0
    assert len(ids) == len(samples) == 1


def test_duplicate_id_rejected(tmp_path):
    record = {
        "id": "r00:d000:m0000",
        "pid": "r00",
        "day": 0,
        "minute": 0,
        "input": "time: Mon 00:00\n",
        "output": "next: kitchen\ndo: []\n",
    }
    path, sha = make_file(tmp_path, [record, record])
    with pytest.raises(ValueError, match="duplicate"):
        scan(path, HOUSE, expected_hash=sha, expected_count=2)


def test_hash_mismatch_rejected(tmp_path):
    record = {
        "id": "r00:d000:m0000",
        "pid": "r00",
        "day": 0,
        "minute": 0,
        "input": "time: Mon 00:00\n",
        "output": "next: kitchen\ndo: []\n",
    }
    path, _ = make_file(tmp_path, [record])
    with pytest.raises(ValueError, match="SHA-256"):
        scan(path, HOUSE, expected_hash="0" * 64, expected_count=1)
