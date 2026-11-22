# Step 2 — Synthetic Resident Behavior and Personalization

**Project:** Smart House SLM / SiR.AI  
**Research role:** Generate heterogeneous daily routines and observable evidence of hidden preferences  
**Status:** Implemented; behavior parameters are synthetic.

## Research objective

Create repeatable but heterogeneous resident trajectories so that a model must infer individualized automation preferences from sensor observations and past manual actions. The resident generator is not trained on measured human behavior.

## Persona construction

`engine/resident.py` defines a `Persona` with work/routine attributes, wake/return/bed anchors, variability and weekend shifts. Personalization variables include preferred **home thermostat setpoints (approximately 19–23.5 °C)**, **away setpoints (approximately 16–18.5 °C)**, coffee habits, and behavioral probabilities `p_react` and `p_forget`.

`sample_persona()` uses seeded pseudo-random sampling. The resulting preference values are **privileged simulator variables**; they are not copied into the eventual model input. The aim is to make these values indirectly inferable through repeated manual behavior, not to give the model the answers.

## Routine generation

Each day combines recognizable temporal anchors (sleep, waking, work departure/return where applicable, bedtime) with stochastic intermediate activities. Activity durations use gamma-distributed draws and a minimum simulation-tick duration, yielding a semi-Markov-like routine rather than independent activity sampling at each tick. `day_timeline()` constructs the sequence; `lookup()` determines current and next activity and time to the next transition.

Activities map to rooms and include periods outside the home. Weekend shifts and resident-specific routine parameters introduce variation while retaining predictable patterns. The generated schedule is synthetic ground truth, not an observed human calendar.

## Manual habits and simulation cycle

Resident actions supply the behavioral signal needed for personalization: manual thermostat adjustments, lighting or device changes, and occasional omissions or delays. `preferred_state()` defines the resident's target device configuration given activity, room, time and persona. Manual interventions provide historical clues about these preferences.

The main `simulate()` loop runs every 10 minutes. It constructs the current routine context, captures a **pre-decision snapshot**, obtains oracle labels when a teacher is present, optionally applies them, executes resident manual behavior, advances thermal dynamics, and records the resulting state and energy estimate. The pre-decision snapshot includes device state, indoor/outdoor temperature, room and door event; it must not be confused with the post-tick row state.

The simulation supports **open-loop** (`follow=False`, record oracle commands only) and **teacher-in-loop** (`follow=True`, execute them). The same routine/weather streams can be paired across these modes, while device states and later manual actions can diverge.

## Splits and validation

`split_residents()` deterministically partitions resident IDs. The research design uses **30 residents: 20 training, 4 validation and 6 held-out residents**. The separation is by resident, not by randomly selected ticks, to reduce leakage from near-duplicate observations and repeated individual preferences.

Tests cover resident determinism, day structure, inferability of preferences from history, split disjointness, and pairing of routine/weather across simulation modes. The latest reported repository test run passed **46 tests** overall.

## Limitations

- Resident preferences and activity probabilities were invented rather than estimated from real households.
- Single-resident occupancy and simplified routines exclude many household interactions.
- Predictability and habit signals may be stronger or more regular than in human behavior.
- Even a resident-disjoint synthetic test set cannot demonstrate real-world personalization.
- The oracle has access to hidden persona/schedule variables that a deployed model would not have.

**Paper-use summary:** A seeded, semi-Markov-like generator creates diverse household trajectories with latent preferences manifested through manual actions. Resident-level holdout is used to study synthetic generalization to unseen personas.

**Primary files:** `engine/resident.py`, `tests/test_engine.py`.  
**Related specification:** `PROTOCOL.md` §§3–4.
