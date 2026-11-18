"""Teacher v1 (Step 3): o gabarito (labels) dos dados.

O teacher é um ORÁCULO: usa informação privilegiada (preferências ocultas do morador e a agenda
verdadeira) que o modelo de ML NÃO vê. O modelo precisa inferir isso da observação + histórico.
Label de cada tick = (próximo cômodo, ações que levam os dispositivos ao estado preferido).
O teacher respeita ajustes manuais recentes do termostato (override hold de 60 min).
"""
import dataclasses
import hashlib
import json
import os
import random
import statistics

from world import apply, initial_state, load_house, parse_action
from resident import INACTIVE, preferred_state, sample_persona, simulate


def to_action(eid: str, v):
    return parse_action(f"{eid}={v:g}" if isinstance(v, float) else f"{eid}={v}")


def oracle(house, p, ctx: dict, state: dict) -> list:
    """Ações (formato compacto, ordenadas) que levam `state` ao estado preferido do morador.
    ctx: act, room, hour, next_act, mins_to_next, held (dispositivos mexidos à mão há pouco)."""
    soon = ctx["act"] in INACTIVE and ctx["next_act"] not in INACTIVE and ctx["mins_to_next"] <= 30
    want = preferred_state(p, house, ctx["act"], ctx["room"], ctx["hour"], soon)
    held = {e for e in ctx.get("held", ()) if house.devices[e]["domain"] == "climate"}   # hold só no termostato
    return sorted(to_action(e, v).compact() for e, v in want.items() if state[e] != v and e not in held)


def population_teacher(house, p, ctx: dict, state: dict) -> list:
    """Referência SEM os segredos: setpoints 21/17 e sem café. Conhece a rotina, não as preferências."""
    q = dataclasses.replace(p, setpoint_home=21.0, setpoint_away=17.0, coffee=False)
    return oracle(house, q, ctx, state)


def ctx_of(row: dict) -> dict:
    return {"act": row["act"], "room": row["room"], "hour": row["minute"] // 60, "next_act": row["next_act"],
            "next_room": row["next_room"], "mins_to_next": row["mins_to_next"], "held": []}


if __name__ == "__main__":
    house = load_house(os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_v1.yaml"))
    rng = random.Random(0)
    personas = [sample_persona(rng, i) for i in range(30)]
    N, per_day = 21, 1440 // house.tick

    # 1) Unidade: o oráculo converge e é idempotente, e respeita override manual
    st = initial_state(house)
    ctx = dict(act="relax", room="living", hour=19, next_act="washing", next_room="bathroom", mins_to_next=40, held=[])
    acts = oracle(house, personas[0], ctx, st)
    print(f"ações do oráculo no estado inicial (r00 gosta de {personas[0].setpoint_home} °C):", acts)
    for a in acts:
        st = apply(house, st, parse_action(a))
    assert oracle(house, personas[0], ctx, st) == []                                   # idempotente
    assert not any(a.startswith("climate") for a in oracle(house, personas[0], {**ctx, "held": ["climate.house"]},
                                                           initial_state(house)))      # respeita override

    # 2) Dois modos de gerar dados, com a MESMA vida do morador (comparação pareada)
    modes = {"open_loop": dict(teacher=oracle, follow=False), "teacher_in_loop": dict(teacher=oracle, follow=True)}
    logs = {m: {p.pid: simulate(house, p, N, seed=7, **kw) for p in personas} for m, kw in modes.items()}
    for p in personas:
        a, b = logs["open_loop"][p.pid], logs["teacher_in_loop"][p.pid]
        assert [(r["act"], r["t_out"]) for r in a] == [(r["act"], r["t_out"]) for r in b]
    print("vida do morador idêntica entre os modos: OK")

    # 3) Estatísticas dos labels e do mundo em cada modo
    days = len(personas) * N
    for mode, per in logs.items():
        rows = [r for p in personas for r in per[p.pid]]
        nonidle = [r for r in rows if r["label_do"]]
        repeats = sum(1 for p in personas for a, b in zip(per[p.pid], per[p.pid][1:]) if a["label_do"] and a["label_do"] == b["label_do"])
        doors = sum(r["act"] == "away" and r["state"]["lock.front_door"] == "unlocked" for r in rows)
        lights = sum(any(v == "on" and e.split(".")[1] != r["room"] for e, v in r["state"].items() if e.startswith("light.")) for r in rows)
        kwh = sum(r["wh"] for r in rows) / 1000 / days
        manual = sum(len(r["manual"]) for r in rows) / days
        print(f"[{mode:>15}] linhas com ação: {len(nonidle)/len(rows):.1%} | ações/linha (com ação): "
              f"{statistics.mean(len(r['label_do']) for r in nonidle):.2f} | labels repetidos/dia: {repeats/days:.1f} | "
              f"manuais/dia: {manual:.1f} | porta destrancada c/ morador fora (ticks/dia): {doors/days:.2f} | "
              f"luz em cômodo vazio (ticks/dia): {lights/days:.2f} | energia: {kwh:.1f} kWh/dia")

    # 4) Headroom de personalização: uma regra que conhece a rotina mas não os segredos acerta quanto?
    pmap = {p.pid: p for p in personas}
    for mode in logs:
        hit = tot = hit_all = tot_all = 0
        for pid, rows in logs[mode].items():
            prev = initial_state(house)
            for r in rows:
                lab, pop = r["label_do"], population_teacher(house, pmap[pid], ctx_of(r), prev)
                tot_all += 1
                hit_all += pop == lab
                if lab:
                    tot += 1
                    hit += pop == lab
                prev = r["state"]
        print(f"[{mode:>15}] regra sem os segredos vs oráculo: exact match em linhas com ação = {hit/tot:.1%} | em todas as linhas = {hit_all/tot_all:.1%}")

    # 5) Determinismo e hash dos labels
    a = simulate(house, personas[0], 14, seed=3, teacher=oracle)
    b = simulate(house, personas[0], 14, seed=3, teacher=oracle)
    assert [r["label_do"] for r in a] == [r["label_do"] for r in b]
    print("hash dos labels (r00, 14 dias, seed 3):", hashlib.sha256(json.dumps([r["label_do"] for r in a]).encode()).hexdigest()[:12])
    print("OK")
