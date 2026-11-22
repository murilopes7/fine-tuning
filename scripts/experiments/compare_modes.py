"""Step 5: Paired open-loop vs teacher-in-loop simulation comparison.

Run from the repository root:

    python -m scripts.experiments.compare_modes \
        --seed 0 \
        --resident-index 0 \
        --days 7 \
        --out data/diagnostics/mode_comparison.json

This experiment does not modify existing datasets or manifests.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from engine.resident import sample_persona, simulate
from engine.teacher import oracle
from engine.world import load_house, apply, parse_action


# Locate the project root independently of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOUSE = PROJECT_ROOT / "engine" / "house_v1.yaml"


def metrics(rows, house):
    """Calculate automation and repetition metrics for one simulation."""

    action_rows = 0
    repeated_full = 0
    repeated_set = 0
    repeated_any = 0
    repeated_instances = 0

    total_instances = 0
    state_changes = 0
    noops = 0

    persistent_next = 0
    persistent_den = 0
    manual = 0

    prev = None
    prev_actions = set()

    for i, row in enumerate(rows):
        actions = set(row["label_do"])
        output = (row["next_room"], tuple(sorted(actions)))

        # Count repeated teacher recommendations.
        if actions:
            action_rows += 1

            if prev is not None and output == prev:
                repeated_full += 1

            if prev is not None and actions == prev_actions:
                repeated_set += 1

            common = actions & prev_actions
            repeated_instances += len(common)

            if common:
                repeated_any += 1

        # Check whether teacher commands would change the decision state.
        state = row["decision"]["state"]

        for action in row["label_do"]:
            total_instances += 1

            new_state = apply(house, state, parse_action(action))

            if new_state != state:
                state_changes += 1
            else:
                noops += 1

            state = new_state

        # Check whether recommended states persist until the next tick.
        if i + 1 < len(rows):
            next_row = rows[i + 1]

            if row["pid"] == next_row["pid"]:
                for action in row["label_do"]:
                    entity, _ = action.split("=", 1)

                    desired_state = apply(
                        house,
                        row["decision"]["state"],
                        parse_action(action),
                    )

                    actual_next_state = next_row["decision"]["state"]

                    persistent_den += 1

                    if actual_next_state[entity] == desired_state[entity]:
                        persistent_next += 1

        manual += len(row["manual"])

        prev = output
        prev_actions = actions

    n = len(rows)
    ticks_per_day = 1440 // house.tick
    simulated_days = n / ticks_per_day

    def fraction(numerator, denominator):
        return round(numerator / denominator, 6) if denominator else 0

    return {
        "ticks": n,
        "action_rows": action_rows,
        "action_row_rate": fraction(action_rows, n),
        "consecutive_identical_full_outputs": repeated_full,
        "consecutive_identical_full_outputs_per_action_row": fraction(
            repeated_full, action_rows
        ),
        "consecutive_identical_action_sets": repeated_set,
        "consecutive_identical_action_sets_per_action_row": fraction(
            repeated_set, action_rows
        ),
        "action_rows_sharing_any_command_with_previous_tick": repeated_any,
        "action_rows_sharing_any_command_rate": fraction(repeated_any, action_rows),
        "repeated_individual_command_instances": repeated_instances,
        "repeated_individual_command_fraction": fraction(
            repeated_instances, total_instances
        ),
        "total_command_instances": total_instances,
        "commands_that_would_change_decision_state": state_changes,
        "commands_already_satisfied_at_decision": noops,
        "next_tick_target_persistence_rate": (
            fraction(persistent_next, persistent_den) if persistent_den else None
        ),
        "manual_actions_per_day": round(manual / simulated_days, 4),
        "simulated_kwh_per_day": round(
            sum(row["wh"] for row in rows) / 1000 / simulated_days,
            4,
        ),
    }


def compare(house_path, seed=0, resident_index=0, days=7):
    """Run paired simulations using the same resident and random seed."""

    house_path = Path(house_path)

    if days < 1:
        raise ValueError("days must be positive")

    if resident_index < 0:
        raise ValueError("resident index must be nonnegative")

    if not house_path.is_file():
        raise FileNotFoundError(f"House configuration not found: {house_path}")

    house = load_house(str(house_path))

    # Generate the same persona for both simulation modes.
    rng = random.Random(seed)
    persona = None

    for i in range(resident_index + 1):
        persona = sample_persona(rng, i)

    # Open-loop: teacher recommendations are not applied.
    open_loop_rows = simulate(
        house,
        persona,
        days,
        seed=seed,
        teacher=oracle,
        follow=False,
    )

    print("Open-loop rows:", len(open_loop_rows))
    print("First row keys:", sorted(open_loop_rows[0].keys()))

    missing = [i for i, row in enumerate(open_loop_rows) if "decision" not in row]

    print("Rows missing decision:", len(missing))
    print("First missing indices:", missing[:10])

    # Teacher-in-loop: teacher recommendations are applied.
    teacher_loop_rows = simulate(
        house,
        persona,
        days,
        seed=seed,
        teacher=oracle,
        follow=True,
    )

    expected_ticks = days * (1440 // house.tick)

    assert len(open_loop_rows) == expected_ticks
    assert len(teacher_loop_rows) == expected_ticks

    # Ensure both modes use the same activity, movement and weather.
    open_loop_sequence = [
        (row["act"], row["next_room"], row["t_out"]) for row in open_loop_rows
    ]

    teacher_loop_sequence = [
        (row["act"], row["next_room"], row["t_out"]) for row in teacher_loop_rows
    ]

    assert open_loop_sequence == teacher_loop_sequence, (
        "Resident routine or weather differs between modes"
    )

    report = {
        "purpose": (
            "Exploratory paired simulation-mode comparison; "
            "not held-out model evaluation"
        ),
        "seed": seed,
        "resident_index": resident_index,
        "pid": persona.pid,
        "days": days,
        "world_sha256": hashlib.sha256(house_path.read_bytes()).hexdigest(),
        "same_activity_next_room_weather": True,
        "open_loop": metrics(open_loop_rows, house),
        "teacher_in_loop": metrics(teacher_loop_rows, house),
        "notes": [
            (
                "Teacher-in-loop applies recommendations before manual "
                "habits. Manual behavior can differ because the state changes."
            ),
            (
                "Repeated commands measure label repetition, not necessarily "
                "unnecessary real-world device calls."
            ),
            ("Commands already satisfied at decision time measure oracle idempotence."),
            (
                "Next-tick target persistence is not a direct measure "
                "of successful execution."
            ),
            (
                "Energy is a synthetic simulator quantity, not calibrated "
                "real-world energy consumption."
            ),
            (
                "This one-resident pilot must not be used to select a "
                "winning mode based on Test A, B or C."
            ),
        ],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--house",
        type=Path,
        default=DEFAULT_HOUSE,
        help="Path to the house YAML configuration",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resident-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostics/mode_comparison.json"),
    )

    args = parser.parse_args()

    report = compare(
        house_path=args.house,
        seed=args.seed,
        resident_index=args.resident_index,
        days=args.days,
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.out),
                "open_loop": report["open_loop"],
                "teacher_in_loop": report["teacher_in_loop"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
