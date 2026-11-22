import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from scripts.experiments.analyze_repetition import analyze
from engine.world import load_house

HOUSE = Path(__file__).resolve().parents[1] / "engine" / "house_v1.yaml"


def test_adjacent_repetition(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = [
        {
            "pid": "r00",
            "day": 0,
            "minute": 0,
            "output": "next: bedroom\ndo: [climate.house=23]\n",
        },
        {
            "pid": "r00",
            "day": 0,
            "minute": 10,
            "output": "next: bedroom\ndo: [climate.house=23]\n",
        },
        {
            "pid": "r00",
            "day": 0,
            "minute": 30,
            "output": "next: bedroom\ndo: [climate.house=23]\n",
        },
    ]
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    r = analyze(p, load_house(str(HOUSE)))
    assert r["counts"]["action_rows"] == 3
    assert r["counts"]["repeated_full_outputs"] == 1
    assert r["counts"]["repeated_command_instances"] == 1
    assert r["counts"]["nonadjacent_comparisons_skipped"] == 1
