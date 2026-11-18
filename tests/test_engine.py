"""Testes do engine (Steps 1 a 3). Rodar na raiz do repositório: pytest -q"""
import os
import random
import statistics

import pytest

from world import apply, initial_state, legal_actions, load_house, observe, parse_action, scene_json, thermal_step, thermostat
from resident import sample_persona, simulate, split_residents
from teacher import oracle

HOUSE_PATH = os.path.join(os.path.dirname(__file__), "..", "engine", "house_v1.yaml")
GOLDEN_WORLD_HASH = "052f8bd9f28c"      # se o mundo mudar: incremente world_version, atualize este hash e o PROTOCOL.md
N_DAYS = 14


@pytest.fixture(scope="module")
def house():
    return load_house(HOUSE_PATH)


@pytest.fixture(scope="module")
def personas():
    rng = random.Random(0)
    return [sample_persona(rng, i) for i in range(30)]


@pytest.fixture(scope="module")
def logs(house, personas):
    return {mode: {p.pid: simulate(house, p, N_DAYS, seed=7, teacher=oracle, follow=follow) for p in personas}
            for mode, follow in (("open", False), ("loop", True))}


# ---------- Step 1: mundo ----------

def test_world_is_versioned(house):
    assert house.world_hash == GOLDEN_WORLD_HASH, "o mundo mudou: incremente world_version e atualize o hash"


def test_action_roundtrip_and_count(house):
    acts = legal_actions(house, initial_state(house))
    assert len(acts) == 26
    assert all(parse_action(a.compact()) == a for a in acts)


@pytest.mark.parametrize("bad", ["climate.house=40", "climate.house=21.2", "light.bedroom=blue", "fan.x=on"])
def test_illegal_actions_are_rejected(house, bad):
    with pytest.raises(ValueError):
        apply(house, initial_state(house), parse_action(bad))


def test_thermostat_keeps_temperature_near_setpoint(house):
    t, heating, trace = 17.0, False, []
    for _ in range(144):
        heating = thermostat(t, 21.0, heating)
        t = thermal_step(t, 6.0, heating, house.tick)
        trace.append(t)
    assert all(19.5 <= x <= 22.5 for x in trace[18:])


def test_observe_is_deterministic_and_missing_rate(house):
    truth = {"room": "kitchen", "t_in": 20.0, "t_out": 6.0, "door_open": False}
    assert observe(house, truth, random.Random(1), p_missing=0.3) == observe(house, truth, random.Random(1), p_missing=0.3)
    rng, miss, total = random.Random(2), 0, 0
    for _ in range(300):
        o = observe(house, truth, rng, p_missing=0.5)
        miss += sum(v is None for v in o.values())
        total += len(o)
    assert 0.4 < miss / total < 0.6


def test_scene_json_has_all_devices(house):
    scene = scene_json(house, initial_state(house))
    assert len(scene["devices"]) == len(house.devices) and scene["world_hash"] == house.world_hash


# ---------- Step 2: moradores ----------

def test_split_by_resident_is_disjoint(personas):
    sp = split_residents(personas)
    ids = sp["train"] + sp["val"] + sp["test"]
    assert len(ids) == len(set(ids)) == 30 and (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (20, 4, 6)


def test_resident_is_deterministic(house, personas):
    p = personas[0]
    assert simulate(house, p, 5, seed=1) == simulate(house, p, 5, seed=1)
    assert simulate(house, p, 5, seed=1) != simulate(house, p, 5, seed=2)


def test_day_structure_and_workers_leave(house, personas, logs):
    per_day = 1440 // house.tick
    for p in personas:
        rows = logs["open"][p.pid]
        for d in range(N_DAYS):
            acts = [r["act"] for r in rows[d * per_day:(d + 1) * per_day]]
            assert acts[0] == "sleep" and acts[-1] == "sleep" and "waking" in acts
            if p.works and d % 7 < 5:
                assert acts.count("away") * house.tick >= 200


def test_preferences_are_inferable_from_history(personas, logs):
    prior = statistics.mean(p.setpoint_home for p in personas)
    errs, errs_prior = [], []
    for p in personas:
        vals = [float(a.split("=")[1]) for r in logs["open"][p.pid] if r["day"] < 7 for a in r["manual"]
                if a.startswith("climate.house=") and float(a.split("=")[1]) >= 19]
        if vals:
            errs.append(abs(statistics.median(vals) - p.setpoint_home))
        errs_prior.append(abs(prior - p.setpoint_home))
    assert statistics.mean(errs) < 0.3 and statistics.mean(errs) < statistics.mean(errs_prior) / 3


# ---------- Step 3: teacher ----------

def test_oracle_is_idempotent(house, personas):
    ctx = dict(act="relax", room="living", hour=19, next_act="washing", next_room="bathroom", mins_to_next=40, held=[])
    st = initial_state(house)
    for a in oracle(house, personas[0], ctx, st):
        st = apply(house, st, parse_action(a))
    assert oracle(house, personas[0], ctx, st) == []


def test_oracle_respects_thermostat_override(house, personas):
    ctx = dict(act="relax", room="living", hour=19, next_act="washing", next_room="bathroom", mins_to_next=40, held=["climate.house"])
    assert not any(a.startswith("climate") for a in oracle(house, personas[0], ctx, initial_state(house)))


def test_life_is_paired_across_modes(logs, personas):
    for p in personas:
        a, b = logs["open"][p.pid], logs["loop"][p.pid]
        assert [(r["act"], r["t_out"]) for r in a] == [(r["act"], r["t_out"]) for r in b]


def test_teacher_in_loop_behaves(logs, personas):
    rows = [r for p in personas for r in logs["loop"][p.pid]]
    days = len(personas) * N_DAYS
    repeats = sum(1 for p in personas for a, b in zip(logs["loop"][p.pid], logs["loop"][p.pid][1:])
                  if a["label_do"] and a["label_do"] == b["label_do"])
    unlocked_away = sum(r["act"] == "away" and r["state"]["lock.front_door"] == "unlocked" for r in rows)
    assert sum(bool(r["label_do"]) for r in rows) / len(rows) < 0.15
    assert repeats / days < 1 and unlocked_away / days < 0.5


def test_open_loop_labels_are_legal(house, personas, logs):
    for p in personas[:5]:
        prev = initial_state(house)
        for r in logs["open"][p.pid]:
            st = prev
            for a in r["label_do"]:
                st = apply(house, st, parse_action(a))            # levanta ValueError se ilegal
            assert (st != prev) == bool(r["label_do"])
            prev = r["state"]
