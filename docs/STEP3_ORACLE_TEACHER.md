# Step 3 — Privileged Oracle for Synthetic Supervision

**Project:** Smart House SLM / SiR.AI  
**Research role:** Generate next-room and device-action labels  
**Status:** Rule-based oracle implemented; label validity is distinct from real-world optimality.

## Research objective

Provide reproducible supervised targets without manually annotating hundreds of thousands of simulated decisions. The oracle is intentionally **privileged**: it can inspect resident preferences and the simulated true routine. The future student model must instead operate on serialized sensor readings, device states and historical manual behavior.

## Inputs and outputs

`engine/teacher.py` implements `oracle(house, persona, ctx, state)`. Context includes current/next activity, room, time to the next activity, and recently manually changed devices. The teacher computes a preferred device configuration and returns commands to move the current state toward it. The supervised output combines a **next-room** target from the routine with a **device-action list** from the oracle.

Actions use the same constrained device vocabulary as the world engine and are emitted in deterministic order. This alignment is essential: a label must be legal under the world definition and parsable by the Step 4 serializer.

## Rule policy

The teacher uses `preferred_state()` to translate hidden preferences, current activity, room and time into desired device states. It compares each desired value with the **pre-decision** device state and recommends a command only when a change is needed. A **30-minute pre-heating lookahead** anticipates imminent returns home. A recent manual thermostat adjustment is protected by an intended **60-minute hold** (six 10-minute ticks), reducing immediate conflict with resident intervention.

The project also includes `population_teacher()`, which substitutes shared temperature/coffee preferences for individual ones. It can serve as a non-personalized reference, although it still has access to the true routine and therefore is not a fully observation-only baseline.

## Label-generation modes

**Open-loop:** Recommendations are recorded as targets but do not change the simulated devices. If the target state remains unsatisfied, the same recommendation may recur on later ticks.

**Teacher-in-loop:** Recommendations are applied before later manual behavior and physical evolution. The state trajectory then changes, often reducing repeated commands. These modes produce **different trajectories and label distributions**, not merely two encodings of identical data. Choice of primary mode remains a research-design question and is treated as an ablation in Step 5.

## Validation and methodological limits

Automated tests exercise oracle idempotence, legal labels, thermostat-override behavior and paired simulation modes. A later seven-day pilot found a large mode-dependent repetition difference, documented in Step 5; this is a property of the synthetic simulation, not proof of improved learned-model performance.

The oracle's rules define the benchmark target, which creates a **circularity risk**: a model can score well by imitating those rules without demonstrating robust household control. Accordingly, planned evaluation includes a no-action baseline, rule-policy baseline, few-shot methods, natural-distribution idle behavior, and closed-loop outcomes. The oracle should not be described as a human expert, empirically optimal controller or realistic sensor-only agent.

**Paper-use summary:** Synthetic labels are produced by a deterministic, privileged rule policy with personalized targets, limited lookahead and respect for recent manual overrides. Evaluation must separate imitation accuracy from closed-loop utility.

**Primary files:** `engine/teacher.py`, `engine/resident.py`, `tests/test_engine.py`.  
**Related specification:** `PROTOCOL.md` §§4–6.
