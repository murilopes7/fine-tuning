# Step 5 — Synthetic Dataset Construction, Integrity and Diagnostic Experiments

**Project:** Smart House SLM / SiR.AI  
**Research role:** Produce supervised data, establish split integrity, and characterize label-distribution risks before model training  
**Status:** Natural/filtered datasets and full structural/temporal-history audits completed; Test C generator and independent verifier prepared, pending local execution.  
**Protocol:** v0.3 candidate, not frozen; advisor and evaluation-harness approval pending.

## 1. Motivation and research questions

The earlier steps establish a versioned smart-home environment (Step 1), seeded synthetic residents with latent preferences (Step 2), a privileged rule-based teacher (Step 3), and a strict, leakage-aware input/output format (Step 4). Step 5 turns those components into a dataset that can support controlled small-language-model experiments.

Generating many examples is not sufficient. The dataset must also preserve temporal ordering, separate residents appropriately, remain reproducible, and avoid teaching a model misleading action frequencies. During preparation, two substantive questions emerged:

1. **Idle/action distribution:** What happens to the target distribution when all action examples are retained but only 5% of idle examples are kept?
2. **Temporal repetition:** Are repeated oracle commands caused by a defective labeler, or by the open-loop design in which the teacher's recommendations do not change the simulated environment?

These are **dataset diagnostics**, not evidence of model accuracy, generalization or energy savings.

## 2. Generation methodology

### 2.1 Unit of observation and labels

The simulator advances in **10-minute ticks** (144 per day). Each tick contains a resident trajectory, sensor-related truth, device state, optional manual interventions, and teacher targets. A supervised example is stored in **JSONL** with a structured text `input` and `output`; the text representation itself is YAML-like.

The input includes time, simulated sensor observations, pre-decision device state and past manual-action history. The output contains `next:` (next room) and `do:` (a canonical list of legal device commands). The teacher sees privileged persona and routine context; those fields are excluded from the student input.

The **pre-decision snapshot** is essential. The simulator saves `decision.state`, `decision.t_in`, `decision.t_out`, `decision.room` and `decision.door_open` before any recommended actions, manual actions or thermal updates for the tick. Serialization uses that snapshot rather than the ordinary post-tick `state`. This establishes a consistent decision boundary between input and target.

### 2.2 Synthetic population and volume

The full reported generation used **30 residents × 80 days × 144 ticks/day = 345,600 examples**. The residents differ in schedule, thermostat preferences, coffee habits and manual reaction/forgetting probabilities. These parameters are sampled, not fitted to empirical household data.

A preliminary **three-resident, two-day pilot** produced **864 examples** (288 per resident). It served as a pipeline smoke test. During full-generation planning, a **six-resident, seven-day benchmark** produced 6,048 examples in 10.373 seconds (approximately 583 examples/s), implying roughly ten minutes for the full example count under a linear extrapolation. This was a runtime estimate, not a measured end-to-end full-run duration.

### 2.3 Split design

The reported full dataset contains:

| Split | Resident/time assignment | Examples |
|---|---|---:|
| Train | 20 residents, days 0–59 | 172,800 |
| Validation | 4 separate residents, days 0–79 | 46,080 |
| Test A | Same 20 training residents, days 60–79 | 57,600 |
| Test B | 6 unseen residents, days 0–79 | 69,120 |
| **Total** | **30 residents; 80 days per resident** | **345,600** |

**Test A** assesses later-day behavior for known synthetic residents; **Test B** assesses generalization to synthetic residents absent from training. Train and Test A intentionally share resident identities but not days. Train, Validation and Test B use disjoint resident groups. This is **not a random row-level split**, which would allow strongly related adjacent ticks to appear on both sides of an evaluation boundary.

The protocol also proposes **Test C** (missing/noisy observations and a layout change) and optional external data, but those results are **not established as completed** in this Step 5 record. Test A/B/C must not be used to select hyperparameters, filtering ratios or data-generation modes before protocol freezing.

### 2.4 Primary generation mode

The full reported dataset was produced in **open-loop mode**: the oracle computes and records device recommendations but they are not executed. Resident manual behavior and physical dynamics still evolve the house. This is a valid observation/labeling convention, but it has consequences: an unexecuted correction may be recommended again at the next tick.

The existing open-loop dataset is retained as the reference artifact. Teacher-in-loop is investigated separately rather than silently substituted into the primary data.

## 3. Filtering, retention and deduplication

### 3.1 Policy and rationale

The draft protocol calls for retaining **every row with at least one teacher action** and a seeded **5% sample of rows with no action**. Filtering operates on previously generated JSONL files; it does not require rerunning the simulator. Original files and their manifests should remain intact, with filtered views written separately.

This design increases the density of actionable examples for supervised training. It also changes the label prior, which could make a model issue unnecessary commands when deployed on a natural stream containing many idle ticks. The policy is therefore an **experimental choice**, not a demonstrated improvement.

### 3.2 Measured filtering counts

| Split | Original rows | Rows with actions (retained) | Idle rows retained | Filtered rows |
|---|---:|---:|---:|---:|
| Train | 172,800 | 114,558 | 2,912 | 117,470 |
| Validation | 46,080 | 32,058 | 701 | 32,759 |
| Test A | 57,600 | 38,647 | 948 | 39,595 |
| Test B | 69,120 | 48,886 | 1,012 | 49,898 |
| **Total** | **345,600** | **234,149** | **5,573** | **239,722** |

For training, the action-bearing proportion changed from **114,558 / 172,800 = 66.3%** to **114,558 / 117,470 = 97.5%**. In other words, only about 2.5% of the filtered training set is idle. This is a major distribution shift and should be reported alongside model results.

The 5% policy is a sampling probability applied to idle rows, not a claim that idle rows will form 5% of the final dataset.

### 3.3 Duplicate handling

The pipeline rejects duplicate **example IDs** and checks that split identifiers do not overlap. It does **not** automatically remove all identical outputs or repeated sensor patterns: distinct 10-minute decisions can legitimately share the same label. Collapsing them indiscriminately would destroy temporal structure and alter the target distribution. Content-level repetition is measured separately.

## 4. Integrity and quality diagnostics

### 4.1 Programmatic checks

The reported diagnostics included:

- SHA-256 manifest checks for dataset files;
- unique example identifiers and no cross-split identifier overlap;
- confirmation that filtered records are subsets of source records;
- confirmation that every source action row is retained;
- YAML parsing and legal target-action validation;
- resident/time split checks.

These checks were reported as passing in the project work log. They are evidence for the checked properties, not a proof that every sensor value or label is semantically correct. A complete independent byte-for-byte comparison of the current local artifacts against an archived pre-change dataset has **not** been established here.

### 4.2 Manual sample inspection

Sampled midnight rows for one resident at 00:00, 00:10 and 00:20 produced `next: bedroom` and `do: []`. This provided a basic sanity check of idle formatting and history, but three rows cannot rule out leakage or labeling errors elsewhere.

More importantly, sampled rows at **06:50, 07:00 and 07:10** showed a thermostat still at **17.0 °C**, with the teacher repeatedly recommending `climate.house=23`. This prompted a focused investigation: if the action was merely labeled rather than applied, repetition might be expected from the simulation design.

### 4.3 Initial repetition analysis

An initial scan found **93,767 identical consecutive full outputs among 114,558 action rows (81.9%)** in the training split. That exploratory calculation compared a row's full `next`/`do` output with the preceding output for the same resident, but did not independently verify strict 10-minute adjacency. It also counted repeated **full outputs**, not repeated individual commands or actual device executions.

Consequently, **81.9% is a diagnostic signal, not a measured rate of wasteful actuation**. The subsequent repetition-analysis script was designed to distinguish strict adjacency, full-output equality, action-set equality and repeated command instances. A verified full-training output from that refined script has not been provided in the available record; do not substitute the pilot percentages for a population-wide estimate.

### 4.4 Diagnostic artifacts

The working diagnostic directory includes or was reported to include:

| Artifact | Purpose |
|---|---|
| `data/diagnostics/report.json` | Dataset integrity and distribution diagnostics |
| `data/diagnostics/samples_32.jsonl` | Small set of inspectable serialized examples |
| `data/diagnostics/sample_mode_comparison.json` | Earlier exploratory mode comparison |
| `data/diagnostics/repetition.json` | Repetition-analysis artifact; full numeric contents not reproduced here |
| `data/diagnostics/mode_comparison.json` | Seven-day paired mode experiment reported below |

A later command-line audit showed that the diagnostics and repetition-analysis CLI entry points still had import errors even while all 46 tests passed. Their successful re-execution after correction **has not been confirmed** in the available record. Therefore, passing unit tests must not be interpreted as verification of every CLI workflow.

## 5. Controlled experiment: open-loop vs teacher-in-loop

### 5.1 Experimental question

Is the repeated-action phenomenon mainly associated with recording oracle recommendations without applying them? To isolate this factor, the same simulated resident and seed were evaluated under two action-application rules.

### 5.2 Configuration and pairing

- **Seed:** 0
- **Resident index:** 0
- **Duration:** 7 days = 1,008 ticks per mode
- **Teacher:** same oracle policy
- **Open-loop:** `follow=False`; recommendations are labels only
- **Teacher-in-loop:** `follow=True`; recommendations update device state
- **Paired quantities:** resident routine and weather trajectory
- **Not held constant:** device-state trajectory, later manual interventions, heating and energy

The comparison reads **pre-decision snapshots** to determine whether a command was already satisfied at the moment of recommendation. This matters because the ordinary row state is recorded after actions and thermal evolution. A local verification found `decision` snapshots in **all 1,008 open-loop rows**.

### 5.3 Measured results

The user's local command generated `data/diagnostics/mode_comparison.json` with the following results:

| Measure | Open-loop | Teacher-in-loop |
|---|---:|---:|
| Simulation ticks | 1,008 | 1,008 |
| Action-bearing ticks | 604 | 78 |
| Action-bearing tick rate | 59.92% | 7.74% |
| Consecutive identical full outputs | 486 | 0 |
| Identical full outputs / action tick | 80.46% | 0.00% |
| Consecutive identical action sets | 502 | 0 |
| Identical action sets / action tick | 83.11% | 0.00% |
| Action ticks sharing a command with previous tick | 566 | 1 |
| Shared-command rate / action tick | 93.71% | 1.28% |
| Repeated individual command instances | 676 | 1 |
| Repeated-command fraction | 87.68% | 0.88% |
| Total command instances | 771 | 114 |
| Commands changing decision-time state | 771 | 114 |
| Commands already satisfied at decision | 0 | 0 |
| Next-tick target persistence | 9.47% | 99.12% |
| Manual actions per day | 11.29 | 0.86 |
| Simulated energy per day | 17.80 kWh | 13.83 kWh |

**Metric definitions matter.** Action-row repetition and command-instance repetition use different denominators. “Already satisfied” compares a proposed command with the state **at decision time**; it is not a judgment of whether the command was unnecessary under a different policy. “Next-tick target persistence” measures whether the recommended target is present at the next tick, not a direct instrumented command-execution success rate.

### 5.4 Interpretation

The teacher-in-loop simulation produced **526 fewer action-bearing ticks** than open-loop (604 versus 78), and repeated command instances fell from **87.7%** to **0.9%**. This is consistent with the mechanism hypothesized from the manual sample: in open-loop, the desired device change remains unexecuted, so the oracle can recommend it repeatedly; in teacher-in-loop, the changed state often eliminates that need.

The result also shows that the modes differ beyond the teacher labels. Manual actions dropped from **11.29 to 0.86 per day**, because resident manual behavior responds to the device state. Simulated energy dropped from **17.80 to 13.83 kWh/day** (about **22.3%** relative to the open-loop figure), but this is a property of an **uncalibrated synthetic energy model**. It cannot be reported as measured real-world savings or as causal evidence that a trained model would achieve such savings.

The paired design controls routine and weather, **not** all downstream events: applying commands changes subsequent states, behavior and physical dynamics. The comparison therefore tests an environment intervention, not two label files on identical state sequences.

### 5.5 What the pilot does not establish

- It covers **one resident, one seed and seven days**; no between-resident uncertainty estimate is available.
- It does not compare fine-tuned models or demonstrate better action prediction.
- It does not show whether teacher-in-loop training is preferable for the intended deployment distribution.
- It does not validate real-household comfort, safety or energy efficiency.
- It does not justify replacing the baseline dataset after inspecting exploratory results.

The provisional decision is to **preserve open-loop as the reference dataset** and evaluate teacher-in-loop as a **separately manifested ablation** with matched seeds/personas and explicitly defined evaluation conditions.

## 6. Verification status and reproducibility

The user reported a complete pytest run with **46 passing tests** on **Python 3.14.7 / pytest 9.2.0**, including tests for simulation behavior, dataset building, filtering, diagnostics, serialization and the paired comparison. The run emitted one `anyio`/pytest deprecation warning; it was not a test failure.

The comparison command used for the reported pilot was:

```bash
python -m scripts.experiments.compare_modes \
  --seed 0 \
  --resident-index 0 \
  --days 7 \
  --out data/diagnostics/mode_comparison.json
```

The following commands describe **verification actions**, not all confirmed successful runs:

```bash
python -m pytest tests/ -q
python -m scripts.dataset.build_dataset --help
python -m scripts.dataset.filter_dataset --help
python -m scripts.dataset.diagnose_dataset --help
python -m scripts.experiments.analyze_repetition --help
```

At the last available audit, the first two dataset CLI help commands worked, while `diagnose_dataset` and `analyze_repetition` needed import corrections. The research record should be updated after those commands are rerun successfully. Do not overwrite original datasets merely to test an entry point.

For a paper-ready reproducibility package, archive **configuration, seed, commit ID, world hash, source/filtered manifest hashes, exact command, dependency versions, raw reports and verification logs**. A test suite passing does not prove unchanged byte-level datasets, and a matching row count does not prove unchanged row content.

## 7. Threats to validity

### Synthetic-data realism

Household schedules, hidden preferences, manual interventions and physical constants were designed for experimentation rather than estimated from measured households. Findings may not transfer to real homes or multiple occupants.

### Privileged supervision and circular labels

The teacher knows latent preferences and true activity context. A student model may learn to imitate teacher rules without recovering meaningful human intent. The study therefore requires non-neural baselines, a no-action floor, a rule reference, and closed-loop outcome evaluation in addition to static label matching.

### Selection and action-prior bias

Filtering changes the training action rate from **66.3% to 97.5%**. A model trained on this distribution may overpredict actions on natural data. Report performance on the **natural primary evaluation view** and on the filtered secondary view, including false-positive actions on idle ticks. The optimal idle ratio has not been established.

### Temporal dependence and duplication

Adjacent ticks share routine and device state, producing correlated labels. Random row-level splitting would inflate performance. Resident/day separation helps but does not make observations independent; statistical analysis should respect pairing and the relevant grouping structure.

### Simulation-mode dependence

Open-loop and teacher-in-loop trajectories are different, so naive comparisons of static action-label accuracy may conflate policy quality with changed state distributions. Mode selection must be declared as an ablation and validated under comparable evaluation procedures.

### Incomplete external and model evaluation

No QLoRA-trained model, Test C robustness evaluation, or real-home outcome is reported in this document. The draft protocol requires advisor review and freezing before using held-out model evaluations to support confirmatory claims.

## 8. Remaining research tasks

1. **Complete the command-line audit:** verify every diagnostics and repetition script runs successfully, not merely that its tests import.
2. **Confirm artifact integrity:** compare manifest hashes and deterministic reruns without replacing archived source data; verify temporal boundaries at row level.
3. **Inspect full-dataset repetition:** publish strict-adjacency, full-output, action-set and command-instance metrics with clear denominators.
4. **Replicate mode comparison:** run several residents and seeds, report dispersion and preserve separate outputs/manifests.
5. **Study the idle ratio:** compare filtered and natural-distribution behavior using development data; quantify unnecessary action predictions.
6. **Finalize token budget:** include the exact model tokenizer, chat template and system prompt; check over-length examples.
7. **Complete robustness setup:** generate Test C variants while preserving resident boundaries and freezing evaluation rules.
8. **Implement A0/A1 evaluation baselines** before spending GPU time on QLoRA; obtain advisor approval and freeze `protocol-v1` before held-out model evaluation.

## 9. Relationship to the planned paper

This step supports the paper's **Dataset Construction**, **Experimental Setup**, and **Threats to Validity** sections. The defensible claims at this stage are: a seeded, structured synthetic dataset was generated with resident/time splits; the filtering rule strongly shifts the action prior; and a paired pilot shows substantial mode-dependent label repetition. Claims about model generalization, safe control, or real-world energy efficiency require later experiments.

**Primary code:** `scripts/dataset/build_dataset.py`, `scripts/dataset/filter_dataset.py`, `scripts/dataset/diagnose_dataset.py`, `scripts/experiments/analyze_repetition.py`, `scripts/experiments/compare_modes.py`.  
**Related records:** `STEP5_RESEARCH_JOURNAL.md`, `PROTOCOL.md`, `ROADMAP.txt`, `data/diagnostics/`.

## 10. Final Step 5 audit and candidate evaluation freeze (2026-11-21)

**Checks passed (operator-reported, not independently rerun here):** all 24 JSONL files matched source manifests; repeat filtering was byte-identical; 345,600 original examples had expected counts, unique IDs, resident/day partitioning and exactly 144 distinct ticks per resident-day. All actions were retained in the filtered sets, with idle retained only from the original rows. All inputs parsed with keys `t`, `obs`, `dev`, `hist`; all outputs passed the strict YAML action parser. The timestamp consistency audit processed **4,115,988 serialized manual-history entries** over all **345,600** examples and the operator confirmed its final PASS message. The timestamp audit is necessary but not a proof of unique original event day across multiple weeks.

**Implementation review:** `simulate()` snapshots decision state before teacher/manual execution; `serialize_examples()` builds from the snapshot and extends manual history only after serializing each example; `serialize_input()` rejects history tuples at or after the decision timestamp. The history window consists of **up to 12 most recent manual commands**, *not* 12 ticks. Teacher persona and next-activity fields are privileged labeler context and are not directly serialized into the student input.

**Frozen-boundary candidate:** Train 20 residents days 0–59 (172,800 rows), Validation 4 separate residents days 0–79 (46,080), Test A the Train residents days 60–79 (57,600), Test B 6 held-out residents days 0–79 (69,120). Preserve the original files and hashes. **Natural unfiltered evaluation** is primary for Validation/Test A/Test B/Test C, with action-only and 5%-idle-filtered views secondary. This deliberately supersedes draft protocol v0.2's filtered-primary rule. Main training uses the 5%-idle-filtered Train split; natural Train remains an ablation option. Main data generation remains open-loop; teacher-in-loop is a separate ablation.

**Test C sensor missingness:** `python -m scripts.dataset.build_test_c` derives deterministic 10%, 20%, and 30% sensor masking conditions from existing natural Test B; `python -m scripts.dataset.verify_test_c` checks all case IDs, gold labels, input non-observation fields and manifests. The code does not rerun the simulator; σ=0.3 °C noise already present in Test B is retained, not redrawn. Motion-sensor missingness is only partly observable because `obs.motion` omits both false and missing sensors; this limits interpretation. 20% is the H4 condition. Altered layout is separate and exploratory.

**Statistical plan:** paired, resident-cluster bootstrap (10,000 resamples), matched cases, Holm correction for H1/H2/H4, three training seeds, resident-level breakdowns, and caution with six Test B clusters. No tuning on Test A/B/C.

**Remaining freeze blockers:** local Test C generation and independent audit, evaluation harness A0/A1 smoke test, validation-only prompt selection, final safety/outcome definitions and advisor approval. A normal Git checkpoint commit is appropriate now; **do not create the `protocol-v1` tag** or claim complete evaluation protocol approval yet.
