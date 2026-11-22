import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from scripts.experiments.compare_modes import compare, metrics

HOUSE = Path(__file__).resolve().parents[1] / "engine" / "house_v1.yaml"


def test_paired_comparison():
    result = compare(HOUSE, seed=0, resident_index=0, days=1)
    assert result["same_activity_next_room_weather"]
    assert result["open_loop"]["ticks"] == 144
    assert result["teacher_in_loop"]["ticks"] == 144
    for mode in ("open_loop", "teacher_in_loop"):
        m = result[mode]
        assert m["action_rows"] <= m["ticks"]
        assert (
            m["repeated_individual_command_instances"] <= m["total_command_instances"]
        )
        assert m["commands_already_satisfied_at_decision"] == 0


def test_deterministic():
    assert compare(HOUSE, days=1) == compare(HOUSE, days=1)
