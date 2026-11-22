# Smart House SLM: Roadmap and Repository Structure

Last updated 2026-11-16 · Overall progress: about 35% (rough estimate, weighted by effort) · Murilo Santos · PIVIC / SiR.AI, UFLA

## How to use this document

Work one step at a time, in the order below. Before leaving a step, run the checkpoint routine at the end. Update the status of each step here when it changes. Status legend: DONE, DRAFT (exists, needs review), NEXT, TODO, PARTIAL.

## Progress at a glance

| Step | Topic | Status | Main files |
| --- | --- | --- | --- |
| 0 | Research question and design | DRAFT | README.md, README.pt-BR.md, PROTOCOL.md |
| 1 | World model | DONE | engine/world.py, engine/house\_v1.yaml |
| 2 | Resident behavior model | DONE | engine/resident.py |
| 3 | Teacher (labels) | DONE | engine/teacher.py |
| 4 | Schema and serialization | NEXT | engine/serialize.py |
| 5 | Data generation and preprocessing | TODO | engine/build\_dataset.py |
| 6 | 3D web view | TODO | web/ |
| 7 | Simulation engine | PARTIAL | engine/ |
| 8 | Fine-tuning | TODO | train/, notebooks/ |
| 9 | Serving and model-in-the-loop | TODO | serve/ |
| 10 | Evaluation | TODO | eval/, results/ |
| 11 | Tracking and reproducibility | PARTIAL | requirements.txt, results/ |
| 12 | Validity, report and presentation | TODO | docs/ |

Suggested order from here: 4, 5, 10 (build baselines A0 to A5 first), 8, then 9 and 7 (closed loop), with 6 in parallel (time-boxed), and 12 at the end.

## Roadmap

### Step 0: Research question and design (DRAFT)

- **Goal:** state falsifiable questions, baselines, metrics and ablations before running experiments.
- **Mine:** hypotheses H1 to H4 with numeric thresholds; arms A0 to A10; splits Train, Validation, Test A, Test B, Test C and optional Test D; paired bootstrap with Holm correction.
- **Alternatives:** reproduce a published result and change one variable; ablation-first design; design-science cycles (build, evaluate, refine).
- **Files:** README.md, README.pt-BR.md, PROTOCOL.md (v0.2 draft).
- **Done when:** the advisor reviewed PROTOCOL.md, the TBD items are filled, and the tag protocol-v1 exists before the first evaluation on any test split.

### Step 1: World model (DONE)

- **Mine:** 4 rooms, 7 devices (4 lights, climate, coffee switch, front-door lock), 7 sensors, 10-minute tick, Home Assistant-style entity ids, a finite action space (26 legal changes from the initial state), a first-order thermal model, and observations with noise and missing readings.
- **Alternatives:** Home Assistant entity/service schema (mine); VirtualHome-style atomic action programs; OpenSHS-style sensor/actuator catalog.
- **Files:** engine/house\_v1.yaml, engine/world.py.
- **Done when:** tests pass and the world\_hash (currently 052f8bd9f28c) is logged in every dataset and run.
- **Caveat:** thermal and energy constants are not calibrated.

### Step 2: Resident behavior model (DONE)

- **Mine:** 30 simulated residents with hidden preferences (home and away setpoints, coffee habit, p\_react, p\_forget); daily routine made of anchors (wake, leave, return, bed) plus semi-Markov activities with gamma durations; manual habits that leak the preferences into the log; split by resident (20 train, 4 validation, 6 test).
- **Alternatives:** semi-Markov chains fitted to real data; human-driven avatar capture (OpenSHS style); LLM-driven generative agents (use offline, then freeze).
- **Files:** engine/resident.py.
- **Done when:** tests pass and preferences are inferable from history (checked by a test). Parameters are invented, not fitted to real data.

### Step 3: Teacher, the labels (DONE)

- **Mine:** a privileged oracle that knows the hidden preferences and the true schedule. Label = next room plus the actions that move devices to the preferred state; 30-minute pre-heat lookahead; 60-minute hold after a manual thermostat change. Two data modes: open-loop and teacher-in-loop.
- **Alternatives:** LLM few-shot teacher (temperature 0, schema validation, discard invalid outputs); human annotation (small gold set); labels learned from real logs.
- **Files:** engine/teacher.py.
- **Done when:** tests pass and the advisor reviewed the definition of 'preferred' (dark hours 07:00 to 18:00, 30-minute pre-heat).

### Step 4: Schema and serialization (NEXT)

- **Goal:** turn simulator rows into the text the model sees (input) and writes (output), with a history window.
- **Mine (draft):** compact YAML input built only from observe() output, device states and a history of recent manual actions; output with next: and do: lines.
- **Alternatives:** JSON function calling; a tiny custom DSL; constrained decoding with a grammar (llama.cpp GBNF, Outlines) so the output is always valid.
- **Files:** engine/serialize.py, tests/test\_serialize.py.
- **Done when:** the token-length report is done, round-trip parsing of outputs works, and the leakage test passes (no privileged field reaches the input).

### Step 5: Data generation and preprocessing (TODO)

- **Mine (plan):** 30 residents times 80 days; days 0 to 59 for train and 60 to 79 for Test A; the resident split gives Test B; Test C adds missing or noisy readings and a layout change; dedupe; keep all action rows and 5% of idle rows; the data mode (open-loop, teacher-in-loop or mix) is an ablation.
- **Alternatives:** simulator replication (OpenSHS style); template-based synthetic data (Home LLM style); mixing in real data.
- **Files:** engine/build\_dataset.py, tests/test\_dataset.py, data/MANIFEST.json.
- **Done when:** every test split has at least 1,000 rows, the manifest has hashes, and splits are by resident or day, never by row.

### Step 6: 3D web view (TODO)

- **Goal:** an instrument for debugging, error replay and demos. It only reads logged states and never changes simulator state. Time-box it to about 15% of total effort.
- **Mine (settings):** Three.js with OrbitControls; perspective FOV about 45 degrees, damping on, maxPolarAngle about 80 degrees, pixel ratio capped at 2; one directional light (2048 PCFSoft shadow map) plus a hemisphere light; one PointLight per lamp driven by device state; the resident as a capsule or glTF moving along the room graph; InstancedMesh for repeated props; exact library versions pinned.
- **Alternatives:** Babylon.js; PlayCanvas; Unity WebGL.
- **Files:** web/ (index.html, src/, public/scene.json exported by world.scene\_json).
- **Done when:** it replays a logged day with a timeline scrubber and shows model and teacher actions side by side.

### Step 7: Simulation engine, single source of truth (PARTIAL)

- **Done:** headless seeded engine, paired random streams, teacher hook.
- **Missing:** model-in-the-loop interface, safety guard, event-log replay, and retiring the old duplicated JS logic.
- **Mine:** fixed time step, seeded RNG, append-only event log, replayable runs.
- **Alternatives:** discrete-event (SimPy); agent-based (Mesa); ECS or game loop.
- **Done when:** one engine feeds data generation, evaluation and the web view.

### Step 8: Fine-tuning, the core (TODO)

- **Mine (arm A6):** 4-bit NF4 base, LoRA r=16 alpha=16 on all linear layers, dropout 0, lr 2e-4, 1 epoch, effective batch 16, 8-bit AdamW, linear schedule, loss on the answer only, max sequence about 384. Base models: Llama-3.2-3B-Instruct (main) and Qwen3-4B-Instruct-2507.
- **Alternatives:** full fine-tuning of a small model; LoRA variants (DoRA, rsLoRA, higher rank) with a DPO or GRPO second stage, or (IA)3 and prompt tuning; no training at all (few-shot or RAG), which is a mandatory baseline.
- **Files:** train/qlora.py, train/configs/\*.yaml, notebooks/ (Colab).
- **Done when:** a first run on a small subset trains, logs a results row, and the adapter is saved to Drive or the Hub.
- **Note:** build baselines A0 to A5 (Step 10) before spending GPU time here.

### Step 9: Serving and model-in-the-loop (TODO)

- **Mine:** GGUF q4\_k\_m, temperature 0, max\_new\_tokens 48, grammar-constrained output, and a rule-based guard that rejects unsafe actions.
- **Alternatives:** llama.cpp or Ollama; vLLM; in-browser WebGPU runtimes.
- **Files:** serve/server.py, serve/guard.py.
- **Done when:** the simulator can call the model and latency is measured.

### Step 10: Evaluation (TODO)

- **Mine:** exact action match (primary), next-room accuracy, macro-F1, format validity, safety violation rate, outcome success (closed-loop), resilience and efficiency; 3 seeds, greedy decoding, paired bootstrap (10,000 resamples), Holm correction; ablations on data size, rank, history window, few-shot k and method.
- **Alternatives:** static held-out accuracy; closed-loop rollouts; user study.
- **Files:** eval/metrics.py, eval/bootstrap.py, eval/baselines.py, eval/run\_eval.py, eval/prompts/, results/runs.csv, figures/.
- **Done when:** arms A0 and A1 run end to end and reproduce the expected values.

### Step 11: Tracking and reproducibility (PARTIAL)

- **Done:** requirements.txt, world\_hash, and the results schema.
- **Missing:** run logger, dataset card, model card.
- **Mine:** log config, seed, git hash, world\_hash, data\_hash, library versions and raw model outputs for every run.
- **Alternatives:** Weights & Biases; MLflow; TensorBoard plus DVC.

### Step 12: Validity, report and presentation (TODO)

- **Mine:** error analysis (20 sampled mistakes per arm), Test D on real data if available, limitations, the technical report and the CIUFLA presentation.
- **Alternatives:** domain randomization; a small real-data fine-tune; a human-subject study.
- **Files:** docs/ (report, dataset card, model card).
- **Main threat:** a model trained on rule-generated data mostly learns the rules.

## Repository structure (target)

&#91;EXISTS\] marks what has already been created. Everything else is planned.

```
smart-house-slm/
|-- README.md                  [EXISTS] front page (English)
|-- README.pt-BR.md            [EXISTS] front page (Portuguese)
|-- PROTOCOL.md                [EXISTS, DRAFT] frozen later with protocol-v1
|-- ROADMAP.txt                [EXISTS] plain-text roadmap
|-- requirements.txt           [EXISTS]
|-- pytest.ini                 [EXISTS]
|-- .gitignore                 [EXISTS]
|-- .github/workflows/ci.yml   [EXISTS] pytest on Python 3.10 and 3.12
|-- engine/
|   |-- house_v1.yaml          [EXISTS] Step 1
|   |-- world.py               [EXISTS] Step 1
|   |-- resident.py            [EXISTS] Step 2
|   |-- teacher.py             [EXISTS] Step 3
|   |-- serialize.py           Step 4
|   `-- build_dataset.py       Step 5
|-- tests/
|   |-- test_engine.py         [EXISTS] 18 tests (Steps 1 to 3)
|   |-- test_serialize.py      Step 4 (includes leakage test)
|   `-- test_dataset.py        Step 5
|-- data/                      git-ignored, except MANIFEST.json and sample/
|   |-- MANIFEST.json          committed: parameters and hashes
|   |-- sample/                committed: about 200 example rows
|   `-- train.jsonl, val.jsonl, test_a.jsonl, test_b.jsonl, test_c.jsonl   not in Git
|-- train/                     qlora.py, configs/
|-- notebooks/                 Colab notebooks
|-- eval/                      metrics.py, bootstrap.py, baselines.py, run_eval.py, prompts/
|-- serve/                     server.py, guard.py
|-- web/                       3D viewer (reads logs only)
|-- results/                   runs.csv (one row per arm, seed and split)
|-- figures/
|-- docs/                      report, dataset card, model card
`-- legacy/                    old notebook and 2D JS demo (not used)
```

### What goes into Git and what does not

- **In Git:** code, tests, configs, house\_v1.yaml, PROTOCOL.md, results/\*.csv, figures, data/MANIFEST.json and data/sample/.
- **Not in Git:** full datasets, model weights (.safetensors, .gguf, .bin), adapters (push them to the Hugging Face Hub or Drive), .env files and any secrets.

### Privileged fields (never allowed in the model input)

act, room, next\_act, next\_room, mins\_to\_next, label\_do, auto, the noise-free t\_in and t\_out, and all persona parameters. The model may only see observe() output, device states at decision time, time of day, and the history of recent manual actions.

## Data schemas

### Simulation log row (JSONL, one row per resident per 10-minute tick)

pid, day, minute, act, room, next\_act, next\_room, mins\_to\_next, t\_in, t\_out, door\_open, heating, wh, state (entity to value), manual (list of entity=value), label\_do (list), auto (list). Note: state is the device state after the actions of that tick. The state at decision time is the previous row's state.

### Training example (draft, Step 4 will finalize it)

```
input:
t: Mon 18:00
obs: {motion: [living], door: closed, temp_in: 20.4, temp_out: 8.1}
dev: {lights: [], heat: 20.5, coffee: off, lock: unlocked}
hist: [Sun 18:05 climate.house=20.5 manual, ...]

output:
next: kitchen
do: [light.living=on]
```

### results/runs.csv (one row per arm, seed and split)

run\_id, arm, base\_model, seed, split, n, exact, next\_acc, f1\_macro, fmt\_valid, unsafe\_rate, latency\_ms, vram\_gb, trainable\_params, train\_minutes, git\_hash, world\_hash, data\_hash. The README schema lacks world\_hash: add it.

### data/MANIFEST.json (example fields, values are placeholders)

world\_version, world\_hash, tick\_minutes, seed, mode (open\_loop, teacher\_in\_loop or mix), idle\_keep, n\_residents, splits (train, val and test lists of ids), days (train and test\_a ranges), rows per split, data\_hash.

### Run ids

Format: arm, base model, seed and timestamp. Example: A6\_llama3b\_s0\_20261201-1530.

## Git workflow

- One branch per step (for example step4-serialize), merged into main when the checkpoint routine passes.
- Commit messages: Step N: what changed.
- Tags: checkpoint-step3 (now), checkpoint-step4 and so on; protocol-v1 (after advisor approval, before the first test evaluation); results-v1 (after the main grid); paper-v1.
- Start with a private repository. Ask the advisor before making it public, and about the license.

## Checkpoint routine (before leaving every step)

- Run pip install -r requirements.txt and pytest -q: everything green locally.
- GitHub Actions is green.
- Read about 30 rows of real output by hand: do the labels make sense?
- The leakage check still passes.
- PROTOCOL.md TBDs and Deviations are updated.
- git commit, git tag checkpoint-stepN, and push.
- Update the status lines in this document.

## Known open risks

- All simulator parameters are invented, not fitted to real data.
- The advisor has not yet reviewed PROTOCOL.md or the teacher definition.
- The code was tested on Python 3.12 only; CI will test 3.10.
- Nothing has been trained or evaluated yet, so there are no scientific results.
- The old notebook and JS demo are out of sync with the new engine.
