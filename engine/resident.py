"""Modelo do morador v1 (Step 2).

Rotina = âncoras (acordar, sair, voltar, dormir) + semi-Markov entre as âncoras (atividades com
duração gamma). Cada morador (Persona) tem PREFERÊNCIAS OCULTAS (setpoints, café, esquecimento) que
aparecem só como ações manuais no histórico. O modelo de ML nunca vê a Persona, só observações + histórico.
"""

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass

from world import (
    apply,
    energy_wh,
    initial_state,
    load_house,
    parse_action,
    thermal_step,
    thermostat,
)

ACT_ROOM = {
    "sleep": "bedroom",
    "waking": "bedroom",
    "washing": "bathroom",
    "breakfast": "kitchen",
    "cooking": "kitchen",
    "relax": "living",
    "away": "outside",
}
DUR = {
    "waking": (10, 4),
    "washing": (15, 6),
    "breakfast": (20, 8),
    "cooking": (40, 12),
    "relax": (60, 30),
    "errand": (90, 40),
}  # (média, desvio) em minutos, distribuição gamma
INACTIVE = ("away", "sleep")


@dataclass(frozen=True)
class Persona:
    pid: str
    works: bool
    wake: float  # minuto médio do dia em que acorda
    prep: float  # minutos entre acordar e sair (dias de trabalho)
    back: float  # minuto médio em que volta
    bed: float  # minuto médio em que vai dormir
    jitter: float  # σ (min) da rotina
    weekend_shift: float  # atraso médio ao acordar no fim de semana (min)
    setpoint_home: float  # PREFERÊNCIA OCULTA (°C)
    setpoint_away: float  # PREFERÊNCIA OCULTA (°C)
    coffee: bool  # PREFERÊNCIA OCULTA
    p_react: float  # probabilidade de agir manualmente numa rotina
    p_forget: float  # probabilidade de esquecer de desligar luz/cafeteira ou trancar


def sample_persona(rng: random.Random, i: int) -> Persona:
    half = lambda lo, hi: round(rng.uniform(lo, hi) * 2) / 2  # múltiplos de 0,5 °C
    return Persona(
        pid=f"r{i:02d}",
        works=rng.random() < 0.7,
        wake=rng.uniform(330, 480),
        prep=rng.uniform(50, 90),
        back=rng.uniform(1020, 1140),
        bed=rng.uniform(1290, 1400),
        jitter=rng.uniform(15, 40),
        weekend_shift=rng.uniform(30, 120),
        setpoint_home=half(19, 23.5),
        setpoint_away=half(16, 18.5),
        coffee=rng.random() < 0.6,
        p_react=rng.uniform(0.7, 0.95),
        p_forget=rng.uniform(0.05, 0.3),
    )


def split_residents(personas, seed=0, n_val=4, n_test=6) -> dict:
    """Split POR MORADOR (Test B = moradores nunca vistos)."""
    ids = [p.pid for p in personas]
    random.Random(seed).shuffle(ids)
    return {
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test : n_test + n_val]),
        "train": sorted(ids[n_test + n_val :]),
    }


def day_timeline(
    p: Persona, weekend: bool, rng: random.Random, min_dur: float = 10.0
) -> list:
    """Lista [(minuto_de_início, atividade)] de um dia. Toda atividade dura >= min_dur (= 1 tick) para não sumir na amostragem."""
    g = lambda m, s: rng.gammavariate((m / s) ** 2, s * s / m)
    clip = lambda x, lo, hi: max(lo, min(hi, x))
    wake = clip(
        rng.gauss(p.wake + (p.weekend_shift if weekend else 0.0), p.jitter), 240, 720
    )
    bed = clip(rng.gauss(p.bed, p.jitter), wake + 600, 1425)
    leaves = p.works and not weekend
    leave = back = None
    if leaves:
        leave = wake + clip(rng.gauss(p.prep, 8), 35, 120)
        back = min(clip(rng.gauss(p.back, p.jitter), leave + 240, bed - 90), bed - 60)
    tl, t, done = (
        [(0.0, "sleep"), (wake, "waking")],
        wake + max(g(*DUR["waking"]), min_dur),
        set(),
    )
    while t < bed - 20:
        if leaves and "work" not in done and t >= leave:  # âncora: ir trabalhar
            tl.append((t, "away"))
            done.add("work")
            t = back
            continue
        h, opts = t / 60, {"relax": 2.0}  # semi-Markov: pesos por hora e por "já feito"
        if h < 11 and "breakfast" not in done:
            opts["breakfast"] = 4.0
        if h < 11 and "washing" not in done:
            opts["washing"] = 3.0
        if 17 <= h < 21 and "cooking" not in done:
            opts["cooking"] = 5.0
        if not leaves and 10 <= h < 18 and "errand" not in done:
            opts["errand"] = 1.0
        act = rng.choices(list(opts), weights=list(opts.values()))[0]
        d = min(max(g(*DUR[act]), min_dur), bed - 15 - t)
        if leaves and "work" not in done:
            d = min(d, leave - t)  # respeita a âncora de saída
        tl.append((t, "away" if act == "errand" else act))
        done.add(act)
        t += max(d, min_dur)
    tl += [(bed - 15, "washing"), (bed, "sleep")]
    return sorted(tl, key=lambda x: x[0])


def lookup(tl: list, m: float):
    """(atividade atual, próxima atividade, minutos até ela)."""
    i = max(j for j, (s, _) in enumerate(tl) if s <= m)
    if i + 1 < len(tl):
        return tl[i][1], tl[i + 1][1], tl[i + 1][0] - m
    return tl[i][1], "sleep", 1440.0


def preferred_state(
    p: Persona, house, act: str, room: str, hour: int, soon_home: bool
) -> dict:
    """Estado de dispositivos que o morador QUER (usa informação oculta). Base dos labels do Step 3."""
    dark, active = (hour < 7 or hour >= 18), act not in INACTIVE
    st = {}
    for eid, d in house.devices.items():
        dom = d["domain"]
        if dom == "light":
            st[eid] = "on" if (active and dark and d["room"] == room) else "off"
        elif dom == "switch":
            st[eid] = "on" if (p.coffee and act == "breakfast") else "off"
        elif dom == "lock":
            st[eid] = "unlocked" if active else "locked"
        elif dom == "climate":
            st[eid] = p.setpoint_home if (active or soon_home) else p.setpoint_away
    return st


def habits(
    p: Persona, state: dict, prev: str, cur: str, hour: int, rng: random.Random
) -> list:
    """Ações manuais que o morador tende a fazer numa transição de atividade: [(atraso_em_ticks, 'entidade=valor')]."""
    out, dark = [], (hour < 7 or hour >= 18)
    rp, rc = ACT_ROOM[prev], ACT_ROOM[cur]

    def maybe(prob, a):
        if rng.random() < prob:
            out.append((rng.choice((0, 1)), a))

    if rp != rc and rp != "outside" and state[f"light.{rp}"] == "on":
        maybe(1 - p.p_forget, f"light.{rp}=off")  # saiu do cômodo
    if rc != "outside" and dark and cur != "sleep":
        maybe(p.p_react, f"light.{rc}=on")  # entrou no cômodo no escuro
    if cur not in INACTIVE and prev in INACTIVE:  # chegou em casa ou acordou
        sp = min(26.0, max(16.0, p.setpoint_home + rng.choice((0, 0, 0, 0.5, -0.5))))
        maybe(p.p_react, f"climate.house={sp:g}")
        maybe(p.p_react, "lock.front_door=unlocked")
    if cur in INACTIVE and prev not in INACTIVE:  # saiu de casa ou foi dormir
        maybe(p.p_react * 0.8, f"climate.house={p.setpoint_away:g}")
        maybe(1 - p.p_forget, "lock.front_door=locked")
        for eid, v in state.items():  # ronda: apagar as luzes que sobraram
            if eid.startswith("light.") and v == "on":
                maybe(1 - p.p_forget, f"{eid}=off")
    if p.coffee and cur == "breakfast":
        maybe(p.p_react, "switch.coffee=on")
    if (
        p.coffee
        and prev == "breakfast"
        and cur != "breakfast"
        and state["switch.coffee"] == "on"
    ):
        maybe(1 - p.p_forget, "switch.coffee=off")
    return out


HOLD_TICKS = 6  # após um ajuste manual, o teacher não mexe naquele dispositivo por 60 min (respeita overrides)


def simulate(
    house, p: Persona, n_days: int, seed: int = 0, teacher=None, follow: bool = False
) -> list:
    """Vida do morador, uma linha por tick.
    teacher=None: só ações manuais. teacher=f: calcula o label de cada tick (row['label_do']).
    follow=True aplica as ações do teacher (teacher no loop); follow=False só registra (open-loop)."""
    rng_life = random.Random(
        f"{seed}|{p.pid}|life"
    )  # rotina e clima: iguais com ou sem controlador (comparação pareada)
    rng_hab = random.Random(f"{seed}|{p.pid}|habits")
    state, t_in, heating, pending, prev, rows = (
        initial_state(house),
        17.0,
        False,
        [],
        "sleep",
        [],
    )
    last_manual = {}
    per_day = 1440 // house.tick
    for day in range(n_days):
        weekend = day % 7 >= 5
        tl, base = (
            day_timeline(p, weekend, rng_life, house.tick),
            3 + rng_life.random() * 9,
        )
        for k in range(per_day):
            m, gk = k * house.tick, day * per_day + k
            cur, nxt, to_next = lookup(tl, m)

            # Snapshot of the world before the teacher acts.
            decision_state = dict(state)
            decision_t_in = t_in
            decision_t_out = base + 5 * math.sin(2 * math.pi * (m / 1440 - 0.375))

            decision_door_open = (ACT_ROOM[prev] == "outside") != (
                ACT_ROOM[cur] == "outside"
            )

            label, auto = None, []
            if teacher is not None:
                held = sorted(e for e, g in last_manual.items() if gk - g <= HOLD_TICKS)
                ctx = {
                    "act": cur,
                    "room": ACT_ROOM[cur],
                    "hour": m // 60,
                    "next_act": nxt,
                    "next_room": ACT_ROOM[nxt],
                    "mins_to_next": to_next,
                    "held": held,
                }
                label = teacher(
                    house, p, ctx, state
                )  # decide com o estado do fim do tick anterior
                if follow:
                    for a in label:
                        state = apply(house, state, parse_action(a))
                    auto = label
            if cur != prev:
                pending += [
                    (gk + dly, a)
                    for dly, a in habits(p, state, prev, cur, m // 60, rng_hab)
                ]
            manual, keep = [], []
            for due, a in pending:
                if due <= gk:
                    new = apply(house, state, parse_action(a))
                    if new != state:
                        state, manual = new, manual + [a]
                        last_manual[a.split("=")[0]] = gk
                else:
                    keep.append((due, a))
            pending = keep
            t_out = base + 5 * math.sin(2 * math.pi * (m / 1440 - 0.375))
            heating = thermostat(t_in, state["climate.house"], heating)
            t_in = thermal_step(t_in, t_out, heating, house.tick)
            row = {
                "pid": p.pid,
                "day": day,
                "minute": m,
                "act": cur,
                "room": ACT_ROOM[cur],
                "next_act": nxt,
                "next_room": ACT_ROOM[nxt],
                "mins_to_next": round(to_next),
                "t_in": round(t_in, 2),
                "t_out": round(t_out, 2),
                "door_open": (ACT_ROOM[prev] == "outside")
                != (ACT_ROOM[cur] == "outside"),
                "heating": heating,
                "wh": round(energy_wh(house, state, heating, house.tick), 1),
                "state": dict(state),
                "manual": manual,
            }
            if teacher is not None:
                row["label_do"], row["auto"] = label, auto
                row["decision"] = {
                    "state": decision_state,
                    "t_in": decision_t_in,
                    "t_out": decision_t_out,
                    "room": ACT_ROOM[cur],
                    "door_open": decision_door_open,
                }
            rows.append(row)
            prev = cur
    return rows


if __name__ == "__main__":
    import os

    house = load_house(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_v1.yaml")
    )
    rng = random.Random(0)
    personas = [sample_persona(rng, i) for i in range(30)]
    sp = split_residents(personas)
    print(
        "residents:",
        {k: len(v) for k, v in sp.items()},
        "| trabalham:",
        sum(p.works for p in personas),
        "/ 30",
    )

    a, b = (
        simulate(house, personas[0], 14, seed=1),
        simulate(house, personas[0], 14, seed=1),
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(
        b, sort_keys=True
    )  # determinismo com seed
    print(
        "hash do log (r00, 14 dias, seed 1):",
        hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:12],
    )

    N = 21
    logs = {p.pid: simulate(house, p, N, seed=7) for p in personas}
    per_day = 1440 // house.tick
    n_manual = unlocked_away = light_empty = 0
    for p in personas:
        rows = logs[p.pid]
        assert len(rows) == N * per_day
        for d in range(N):
            day = rows[d * per_day : (d + 1) * per_day]
            acts = [r["act"] for r in day]
            assert acts[0] == "sleep", (p.pid, d, acts[:3])  # estrutura do dia
            assert acts[-1] == "sleep", (p.pid, d, acts[-3:])
            assert "waking" in acts, (p.pid, d)
            if p.works and d % 7 < 5:
                assert acts.count("away") * house.tick >= 200  # foi trabalhar
        for r in rows:
            n_manual += len(r["manual"])
            unlocked_away += (
                r["act"] == "away" and r["state"]["lock.front_door"] == "unlocked"
            )
            light_empty += any(
                v == "on" and e.split(".")[1] != r["room"]
                for e, v in r["state"].items()
                if e.startswith("light.")
            )
    days = len(personas) * N
    print(
        f"ações manuais/dia: {n_manual / days:.1f} | ticks/dia com porta destrancada e morador fora: {unlocked_away / days:.2f}"
        f" | ticks/dia com luz acesa em cômodo vazio: {light_empty / days:.2f}"
    )

    # Teste-chave: a preferência oculta é inferível do histórico? (justifica a history window)
    prior = statistics.mean(p.setpoint_home for p in personas)
    for window in (1, 3, 7, 14):
        errs, errs_prior, miss = [], [], 0
        for p in personas:
            vals = [
                float(a.split("=")[1])
                for r in logs[p.pid]
                if r["day"] < window
                for a in r["manual"]
                if a.startswith("climate.house=") and float(a.split("=")[1]) >= 19
            ]
            if vals:
                errs.append(abs(statistics.median(vals) - p.setpoint_home))
            else:
                miss += 1
            errs_prior.append(abs(prior - p.setpoint_home))
        print(
            f"janela {window:>2} dias: MAE histórico = {statistics.mean(errs):.2f} °C ({miss} sem dados) | MAE só com prior = {statistics.mean(errs_prior):.2f} °C"
        )
    print("OK")
