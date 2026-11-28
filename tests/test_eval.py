import json
from pathlib import Path

import pytest

from engine.serialize import serialize_output
from engine.world import load_house
from eval.metrics import evaluate_outputs
from eval.run_eval import run


@pytest.fixture
def house():
    return load_house(str(Path(__file__).resolve().parents[1] / "engine/house_v1.yaml"))


def test_metrics_invalid_and_non_idle(house):
    targets = [
        serialize_output(house, "kitchen", []),
        serialize_output(house, "bedroom", ["light.kitchen=on"]),
    ]
    result = evaluate_outputs(house, targets, [targets[0], "texto inválido"])
    assert result["n"] == 2
    assert result["exact_action_match"] == 0.5
    assert result["next_room_accuracy"] == 0.5
    assert result["format_validity"] == 0.5
    assert result["non_idle_exact_action_match"] == 0


def test_metrics_reject_empty(house):
    with pytest.raises(ValueError):
        evaluate_outputs(house, [], [])


def test_runner_train_only_for_majority(tmp_path, house):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"

    def write(path, targets):
        path.write_text(
            "".join(json.dumps({"output": target}) + "\n" for target in targets)
        )

    train_target = serialize_output(house, "kitchen", ["light.kitchen=on"])
    val_target = serialize_output(house, "bedroom", [])
    write(train, [train_target, train_target])
    write(val, [val_target])
    results = run(
        train, val, Path(__file__).resolve().parents[1] / "engine/house_v1.yaml"
    )
    assert results["A0"]["exact_action_match"] == 1
    assert results["A0b"]["exact_action_match"] == 0
    assert "next: kitchen" in results["A0b_prediction"]


def test_macro_f1_perfect(house):
    target = serialize_output(house, "kitchen", ["light.kitchen=on"])

    result = evaluate_outputs(house, [target], [target])

    assert result["macro_f1"] == 1.0
    assert result["f1_per_class"]["light.kitchen=on"] == 1.0


def test_macro_f1_partial_match(house):
    target = serialize_output(
        house, "kitchen", ["light.kitchen=on", "light.bedroom=off"]
    )

    prediction = serialize_output(house, "kitchen", ["light.kitchen=on"])

    result = evaluate_outputs(house, [target], [prediction])

    assert result["exact_action_match"] == 0.0
    assert result["macro_f1"] == 0.5
    assert result["f1_per_class"]["light.kitchen=on"] == 1.0
    assert result["f1_per_class"]["light.bedroom=off"] == 0.0


def test_macro_f1_false_positive(house):
    target = serialize_output(house, "kitchen", [])

    prediction = serialize_output(house, "kitchen", ["light.kitchen=on"])

    result = evaluate_outputs(house, [target], [prediction])

    assert result["macro_f1"] == 0.0
    assert result["exact_action_match"] == 0.0


def test_macro_f1_invalid_output(house):
    target = serialize_output(house, "kitchen", ["light.kitchen=on"])

    result = evaluate_outputs(house, [target], ["texto inválido"])

    assert result["macro_f1"] == 0.0
    assert result["format_validity"] == 0.0


def test_macro_f1_no_actions(house):
    target = serialize_output(house, "kitchen", [])

    result = evaluate_outputs(house, [target], [target])

    assert result["macro_f1"] is None
    assert result["f1_per_class"] == {}
    assert result["exact_action_match"] == 1.0


def test_metrics_action_order(house):
    target = "next: kitchen\ndo: [light.kitchen=on, light.bedroom=off]\n"

    prediction = "next: kitchen\ndo: [light.bedroom=off, light.kitchen=on]\n"

    result = evaluate_outputs(house, [target], [prediction])

    assert result["exact_action_match"] == 1.0
    assert result["macro_f1"] == 1.0


def test_metrics_different_lengths(house):
    target = serialize_output(house, "kitchen", [])

    with pytest.raises(ValueError):
        evaluate_outputs(house, [target], [])
