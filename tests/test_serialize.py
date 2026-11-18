"""Step 4 tests: run with pytest -q from the repository root."""

import os
import random
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from world import initial_state, load_house, observe
from resident import sample_persona, simulate
from teacher import oracle
from serialize import (
    serialize_input,
    serialize_output,
    parse_output,
    serialize_examples,
    token_length_report,
)


@pytest.fixture(scope="module")
def house():
    return load_house(
        str(Path(__file__).resolve().parents[1] / "engine" / "house_v1.yaml")
    )


@pytest.fixture(scope="module")
def rows(house):
    return simulate(
        house, sample_persona(random.Random(0), 0), 1, seed=7, teacher=oracle
    )


def test_output_roundtrip(house):
    for actions in ([], ["light.living=on"], ["climate.house=21", "switch.coffee=off"]):
        text = serialize_output(house, "kitchen", actions)
        assert (
            serialize_output(
                house,
                **{
                    "next_room": parse_output(house, text)["next"],
                    "actions": parse_output(house, text)["do"],
                },
            )
            == text
        )


@pytest.mark.parametrize(
    "text",
    [
        "next: kitchen\ndo: [light.living=on]\nsecret: true",
        "next: kitchen\ndo: light.living=on",
        "next: attic\ndo: []",
        "next: kitchen\ndo: [climate.house=21.2]",
        "next: kitchen\ndo: [light.living=on, light.living=off]",
        "next: kitchen\ndo: [fan.unknown=on]",
        "next: kitchen\ndo: [climate.house=40]",
        "next: kitchen\nnext: living\ndo: []",
    ],
)
def test_invalid_output(house, text):
    with pytest.raises(ValueError):
        parse_output(house, text)


def test_leakage_whitelist(house, rows):
    # Privileged keys are never serialized, even if attached to source rows.
    row = dict(rows[10], persona={"setpoint_home": 23.5}, future="SECRET_CANARY")
    readings = observe(
        house,
        {
            "room": row["room"],
            "t_in": 20.0,
            "t_out": 8.0,
            "door_open": row["door_open"],
        },
        random.Random(1),
    )
    text = serialize_input(
        house,
        day=row["day"],
        minute=row["minute"],
        readings=readings,
        state=initial_state(house),
    )
    data = yaml.safe_load(text)
    assert set(data) == {"t", "obs", "dev", "hist"}
    assert set(data["obs"]) == {"motion", "door", "temp_in", "temp_out"}
    assert set(data["dev"]) == {"lights", "heat", "coffee", "lock"}
    assert "SECRET_CANARY" not in text and "setpoint_home" not in text
    assert not any(
        key in data
        for key in (
            "act",
            "room",
            "next_act",
            "next_room",
            "mins_to_next",
            "label_do",
            "auto",
            "t_in",
            "t_out",
        )
    )
    with pytest.raises(ValueError):
        serialize_input(
            house,
            day=0,
            minute=100,
            readings={**readings, "act": "sleep"},
            state=initial_state(house),
        )


def test_history_is_past_only(house):
    readings = observe(
        house,
        {"room": "kitchen", "t_in": 20, "t_out": 8, "door_open": False},
        random.Random(1),
    )
    history = [(0, m, "light.kitchen=on") for m in (10, 20, 30)]
    base = dict(day=0, minute=40, readings=readings, state=initial_state(house))
    assert (
        len(
            yaml.safe_load(
                serialize_input(house, **base, manual_history=history, history_window=0)
            )["hist"]
        )
        == 0
    )
    assert (
        len(
            yaml.safe_load(
                serialize_input(house, **base, manual_history=history, history_window=6)
            )["hist"]
        )
        == 3
    )
    with pytest.raises(ValueError):
        serialize_input(
            house, **base, manual_history=history + [(0, 40, "light.kitchen=on")]
        )


def test_examples_use_previous_state_and_no_current_manual(house, rows):
    examples = serialize_examples(house, rows, seed=42)
    assert len(examples) == len(rows)
    assert (
        yaml.safe_load(examples[0]["input"])["dev"]["heat"]
        == initial_state(house)["climate.house"]
    )
    for i in range(1, len(rows)):
        assert (
            yaml.safe_load(examples[i]["input"])["dev"]["heat"]
            == rows[i - 1]["state"]["climate.house"]
        )
    assert all(
        parse_output(house, e["output"])["do"] == sorted(r["label_do"])
        for e, r in zip(examples, rows)
    )
    assert examples == serialize_examples(house, rows, seed=42)


def test_missing_readings(house, rows):
    examples = serialize_examples(house, rows[:3], p_missing=1.0)
    obs = yaml.safe_load(examples[0]["input"])["obs"]
    assert obs == {"motion": [], "door": None, "temp_in": None, "temp_out": None}


def test_token_report(house, rows):
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    report = token_length_report(serialize_examples(house, rows[:10]), DummyTokenizer())
    assert report["n"] == 10
    assert report["input_tokens"]["max"] > 0


def test_decision_state_matches_teacher(house):
    persona = sample_persona(random.Random(0), 0)
    seen_states = []

    def recording_teacher(house, persona, ctx, state):
        seen_states.append(dict(state))
        return oracle(house, persona, ctx, state)

    rows = simulate(
        house,
        persona,
        1,
        seed=7,
        teacher=recording_teacher,
    )

    assert len(seen_states) == len(rows)

    for row, state_at_decision in zip(rows, seen_states):
        assert row["decision"]["state"] == state_at_decision
