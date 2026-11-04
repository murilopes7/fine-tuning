"""Modelo de mundo v1 da casa inteligente.

Fonte única da verdade para: simulador, construtor de dataset, avaliação e viewer 3D.
Estado verdadeiro (truth) != observação (o que o modelo vê): o modelo recebe apenas `observe(...)`.
"""
import hashlib
import json
import random
import re
from collections import deque
from dataclasses import dataclass

import yaml

ENTITY_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")

# Espaço de ações no estilo Home Assistant: domínio -> serviço -> estado resultante
SERVICES = {
    "light":   {"turn_on": "on", "turn_off": "off"},
    "switch":  {"turn_on": "on", "turn_off": "off"},
    "lock":    {"lock": "locked", "unlock": "unlocked"},
    "climate": {"set_temperature": None},          # recebe a temperatura-alvo (value)
}
TEMP_MIN, TEMP_MAX, TEMP_STEP = 16.0, 26.0, 0.5
INITIAL = {"light": "off", "switch": "off", "lock": "locked", "climate": 17.0}


@dataclass(frozen=True)
class Action:
    entity_id: str
    service: str
    value: float | None = None

    def compact(self) -> str:
        """Forma compacta usada nos dados de treino: light.bedroom=on, climate.house=21"""
        dom = self.entity_id.split(".")[0]
        v = SERVICES[dom][self.service] if self.value is None else f"{self.value:g}"
        return f"{self.entity_id}={v}"


def parse_action(text: str) -> Action:
    """Inverso de Action.compact(); levanta ValueError se o texto for inválido."""
    eid, _, v = text.strip().partition("=")
    if not ENTITY_RE.match(eid) or not v:
        raise ValueError(f"ação inválida: {text!r}")
    dom = eid.split(".")[0]
    if dom not in SERVICES:
        raise ValueError(f"domínio desconhecido: {dom}")
    if dom == "climate":
        return Action(eid, "set_temperature", float(v))
    for svc, res in SERVICES[dom].items():
        if res == v:
            return Action(eid, svc)
    raise ValueError(f"valor inválido para {dom}: {v!r}")


class House:
    def __init__(self, raw: dict):
        self.raw = raw
        self.rooms = {r["id"]: r for r in raw["rooms"]}
        self.devices = {d["id"]: d for d in raw["devices"]}
        self.sensors = {s["id"]: s for s in raw["sensors"]}
        self.tick = raw["tick_minutes"]
        validate(self)

    @property
    def world_hash(self) -> str:
        """Hash do mundo: registre em todo run para garantir reprodutibilidade."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]

    def neighbors(self) -> dict:
        g = {r: set() for r in self.rooms} | {"outside": set()}
        for d in self.raw["doors"]:
            a, b = d["between"]
            g[a].add(b)
            g[b].add(a)
        return g


def load_house(path: str) -> House:
    with open(path, encoding="utf-8") as f:
        return House(yaml.safe_load(f))


def validate(h: House) -> None:
    """Invariantes do mundo. Falha cedo, antes de gerar qualquer dado."""
    ids = list(h.devices) + list(h.sensors)
    if len(ids) != len(set(ids)):
        raise ValueError("ids duplicados")
    for eid in ids:
        if not ENTITY_RE.match(eid):
            raise ValueError(f"entity_id inválido: {eid}")
    for r in h.rooms.values():
        if min(r["rect"][2:]) <= 0:
            raise ValueError(f"cômodo com área inválida: {r['id']}")
    for d in h.devices.values():
        if d["domain"] not in SERVICES:
            raise ValueError(f"domínio sem serviços: {d['domain']}")
        if d["id"].split(".")[0] != d["domain"]:
            raise ValueError(f"id e domínio inconsistentes: {d['id']}")
        x, _, z = d["pos"]
        rx, rz, w, dp = h.rooms[d["room"]]["rect"]      # KeyError se o cômodo não existe
        if not (rx <= x <= rx + w and rz <= z <= rz + dp):
            raise ValueError(f"dispositivo fora do cômodo: {d['id']}")
    for s in h.sensors.values():
        if s["room"] not in h.rooms and s["room"] != "outside":
            raise ValueError(f"sensor em cômodo inexistente: {s['id']}")
    g, seen, q = h.neighbors(), {"outside"}, deque(["outside"])
    while q:                                            # toda a casa deve ser alcançável de fora
        for n in g[q.popleft()] - seen:
            seen.add(n)
            q.append(n)
    if seen != set(g):
        raise ValueError(f"cômodos inalcançáveis: {set(g) - seen}")


# ---------- Estado dos dispositivos e espaço de ações ----------

def initial_state(h: House) -> dict:
    return {eid: INITIAL[d["domain"]] for eid, d in h.devices.items()}


def legal_actions(h: House, state: dict, changes_only: bool = True) -> list:
    """Todas as ações válidas. Base para decodificação restrita, guarda de segurança e avaliação."""
    out = []
    for eid, d in h.devices.items():
        dom = d["domain"]
        if dom == "climate":
            n = int(round((TEMP_MAX - TEMP_MIN) / TEMP_STEP)) + 1
            for k in range(n):
                t = TEMP_MIN + k * TEMP_STEP
                if not changes_only or t != state[eid]:
                    out.append(Action(eid, "set_temperature", t))
        else:
            for svc, res in SERVICES[dom].items():
                if not changes_only or res != state[eid]:
                    out.append(Action(eid, svc))
    return out


def apply(h: House, state: dict, a: Action) -> dict:
    """Aplica uma ação e devolve novo estado; levanta ValueError se ilegal."""
    if a.entity_id not in h.devices:
        raise ValueError(f"entidade inexistente: {a.entity_id}")
    dom = h.devices[a.entity_id]["domain"]
    if a.service not in SERVICES[dom]:
        raise ValueError(f"serviço {a.service} inválido para {dom}")
    new = dict(state)
    if dom == "climate":
        v = a.value
        if v is None or not (TEMP_MIN <= v <= TEMP_MAX) or abs(v / TEMP_STEP - round(v / TEMP_STEP)) > 1e-9:
            raise ValueError(f"temperatura fora do domínio: {v}")
        new[a.entity_id] = float(v)
    else:
        new[a.entity_id] = SERVICES[dom][a.service]
    return new


# ---------- Dinâmica física mínima ----------

def thermal_step(t_in, t_out, heating, dt_min, tau_h=6.0, heat_c_per_h=5.0):
    """Modelo térmico de 1ª ordem: perde calor para fora (constante de tempo tau_h) e aquece a heat_c_per_h."""
    dt = dt_min / 60.0
    return t_in + dt * ((t_out - t_in) / tau_h + (heat_c_per_h if heating else 0.0))


def thermostat(t_in, setpoint, heating_prev, band=0.5):
    """Histerese de ±band °C em torno do setpoint (evita liga/desliga a cada tick)."""
    if t_in < setpoint - band:
        return True
    if t_in > setpoint + band:
        return False
    return heating_prev


def energy_wh(h: House, state: dict, heating: bool, dt_min: float) -> float:
    p, w = h.raw["power_w"], 0.0
    for eid, v in state.items():
        dom = h.devices[eid]["domain"]
        if dom in ("light", "switch") and v == "on":
            w += p[dom]
    if heating:
        w += p["climate"]
    return w * dt_min / 60.0


# ---------- Observação parcial e ruidosa (o que o modelo realmente vê) ----------

def observe(h: House, truth: dict, rng: random.Random, noise_c=0.3, p_missing=0.0) -> dict:
    """truth: {'room': str, 't_in': float, 't_out': float, 'door_open': bool}.
    p_missing = fração de leituras ausentes (None): mede resiliência (RQ3)."""
    obs = {}
    for sid, s in h.sensors.items():
        if rng.random() < p_missing:
            obs[sid] = None
        elif s["type"] == "motion":
            obs[sid] = truth["room"] == s["room"]
        elif s["type"] == "contact":
            obs[sid] = bool(truth["door_open"])
        else:
            key = "t_out" if s["room"] == "outside" else "t_in"
            obs[sid] = round(truth[key] + rng.gauss(0, noise_c), 1)
    return obs


# ---------- Exportação para o viewer 3D ----------

def scene_json(h: House, state: dict) -> dict:
    """O viewer 3D só lê este JSON (nunca altera o estado do simulador)."""
    return {
        "world_version": h.raw["world_version"],
        "world_hash": h.world_hash,
        "wall_height": h.raw["wall_height"],
        "rooms": [{"id": r["id"], "type": r["type"], "rect": r["rect"]} for r in h.raw["rooms"]],
        "doors": h.raw["doors"],
        "devices": [{"id": e, "domain": d["domain"], "room": d["room"], "pos": d["pos"], "state": state[e]}
                    for e, d in h.devices.items()],
    }


if __name__ == "__main__":
    import os
    h = load_house(os.path.join(os.path.dirname(__file__), "house_v1.yaml"))
    st = initial_state(h)
    print("world_hash:", h.world_hash, "| rooms:", len(h.rooms), "| devices:", len(h.devices))
    acts = legal_actions(h, st)
    print("ações legais (só mudanças):", len(acts), "| exemplo:", acts[0].compact())
    assert parse_action("climate.house=21").compact() == "climate.house=21"
    assert all(parse_action(a.compact()) == a for a in acts)          # round-trip
    st2 = apply(h, st, parse_action("light.bedroom=on"))
    assert st2["light.bedroom"] == "on" and st["light.bedroom"] == "off"
    for bad in ("climate.house=40", "climate.house=21.2", "light.bedroom=blue", "fan.x=on"):
        try:
            apply(h, st, parse_action(bad))
            raise SystemExit(f"deveria falhar: {bad}")
        except ValueError:
            pass
    # um dia com termostato: setpoint 21, fora a 6 °C
    t, heating, wh = 17.0, False, 0.0
    trace = []
    for k in range(144):
        heating = thermostat(t, 21.0, heating)
        t = thermal_step(t, 6.0, heating, h.tick)
        wh += energy_wh(h, st, heating, h.tick)
        trace.append(t)
    print(f"térmica: min/max após aquecer = {min(trace[18:]):.2f}/{max(trace[18:]):.2f} °C | energia = {wh/1000:.1f} kWh")
    obs1 = observe(h, {"room": "kitchen", "t_in": 20.0, "t_out": 6.0, "door_open": False}, random.Random(0), p_missing=0.2)
    obs2 = observe(h, {"room": "kitchen", "t_in": 20.0, "t_out": 6.0, "door_open": False}, random.Random(0), p_missing=0.2)
    assert obs1 == obs2                                                 # determinismo com seed
    print("observação (20% ausentes, seed 0):", obs1)
    print("scene_json devices:", len(scene_json(h, st)["devices"]))
    print("OK")
