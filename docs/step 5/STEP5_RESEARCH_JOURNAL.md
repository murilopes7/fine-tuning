# Step 5 — Dataset Generation: Research Journal

**Project:** Smart House SLM / SiR.AI  
**Status:** Work in progress; protocol v0.2 is a draft awaiting advisor review.  
**Evidence basis:** User-reported terminal output and examples in the project conversation; uploaded PROTOCOL v0.2 and source files; bundled mode-comparison report. Remote Git state and complete local datasets have not been independently inspected.

## Purpose and reproducibility rules

Record not only successful implementation but also unexpected behavior, attempted fixes, measured results, decisions and unresolved questions. Preserve original dataset hashes; never silently replace held-out data or change protocol decisions after looking at test performance. Freeze `protocol-v1` before held-out model evaluations.

## Chronological work log

### Task 1 — Protocol audit

- **Goal:** Align generated data with README, ROADMAP and PROTOCOL v0.2.
- **Plan:** 30 residents × 80 days × 144 ten-minute ticks = **345,600** examples; resident-disjoint train/validation/test, with Test A on held-out days of train residents.
- **Result:** Audit prepared; final protocol still awaits advisor approval.

### Task 2 — Pilot, performance troubleshooting and protocol splits

- **Pilot:** Three residents over two days yielded 864 examples (288 each for train, validation and test). This was a pipeline smoke test, not the full protocol.
- **Difficulty:** Full dataset generation appeared slow, taking up to about 15 minutes. No crash was demonstrated.
- **Investigation:** Six residents × seven days benchmarked at **6,048 examples in 10.373 seconds**, approximately 583 examples/second. Linear estimate for 345,600 examples was approximately 9 minutes 53 seconds; full-run overhead could be higher.
- **Action:** Continued with protocol-aligned full generation; verified manifest counts using a terminal script.
- **Verified counts:** Train **172,800** (20 residents × 60 days); validation **46,080** (4 × 80); Test A **57,600** (20 × 20 held-out days); Test B **69,120** (6 × 80). Resident-group disjointness check passed.
- **Limit:** Counts and disjoint resident groups alone did not prove row-level temporal integrity.

### Task 3 — Filtering and deduplication

- **Why:** Follow protocol's rule to retain all rows with actions and 5% of idle rows, while preserving unfiltered evaluation distributions.
- **Implementation:** Filter existing JSONL without rerunning the simulator. YAML remains the text format inside each JSONL input/output record; JSONL is the dataset container.
- **Results:**

| Split | Original | Filtered | Original action rows | Filtered idle rows |
|---|---:|---:|---:|---:|
| Train | 172,800 | 117,470 | 114,558 | 2,912 |
| Validation | 46,080 | 32,759 | 32,058 | 701 |
| Test A | 57,600 | 39,595 | 38,647 | 948 |
| Test B | 69,120 | 49,898 | 48,886 | 1,012 |
| **Total** | **345,600** | **239,722** | **234,149** | **5,573** |

- **Unexpected finding:** Train action rate increased from **66.3%** to **97.5%** after filtering. This could bias a model toward unnecessary automation; do not assume filtering improves performance.
- **Testing:** User reported filter unit tests passed; repeated filter run produced matching reported SHA-256 hashes. Direct byte-for-byte comparison between output directories was requested, but its terminal result was not supplied in this conversation.
- **Deduplication policy:** Duplicate IDs are rejected; meaningful repeated observations at distinct ticks are preserved. Document this as a deliberate policy rather than claiming all repeated content was removed.
- **Decision:** Keep original `data/protocol_splits/` untouched and use a separate `data/protocol_filtered/` directory. Compare filtered versus unfiltered training later as an ablation if budget allows.

### Task 4 — Integrity diagnostics and manual label inspection

- **Checks passed (user-reported):** SHA-256 hashes; unique IDs; no cross-split ID overlap; filtered records are subsets of source; all action rows retained; YAML parsing.
- **Manual inspection, idle:** Three consecutive midnight examples (`r01`, day 0, 00:00/00:10/00:20) had `next: bedroom`, `do: []`; no obvious future-label leakage in displayed inputs. This does not establish correctness for all records.
- **Manual inspection, actions:** At 06:50, 07:00, 07:10, the house still showed `heat: 17.0`, while teacher output repeated `do: [climate.house=23]` each tick.
- **Hypothesis:** In open-loop generation, teacher recommendations are recorded but not applied; therefore the teacher can legitimately request the same device change again.
- **Investigation:** A preliminary terminal script compared each action row's full output (`next` and `do`) to the preceding output for the same resident. **93,767 / 114,558 = 81.9%** identical consecutive outputs by that script's definition.
- **Measurement caveat:** The preliminary script did not explicitly verify 10-minute adjacency and compared entire outputs rather than individual commands. It must not be presented as a measured rate of wasteful device actuation.
- **Next action:** Added separate repetition-analysis script to check true adjacency, full action-set repetition and individual-command repetition. Await user's full-data output.

### Controlled simulation-mode comparison

- **Why:** Determine whether the repeated labels stem from open-loop state dynamics rather than a parser or label-generation defect.
- **Design:** One synthetic resident, seven days, seed 0, paired open-loop (`follow=False`) and teacher-in-loop (`follow=True`) simulations. Both use the same resident routine and weather trajectory, but house states can diverge because only the latter applies teacher commands.
- **Pilot results (from bundled sample report):**

| Metric | Open-loop | Teacher-in-loop |
|---|---:|---:|
| Action ticks / 1,008 | 604 | 78 |
| Action rate | 59.9% | 7.7% |
| Repeated consecutive full outputs among action rows | 80.5% | 0.0% |
| Repeated individual commands among command instances | 87.7% | 0.9% |
| Commands already satisfied at decision time | 0 | 0 |
| Recommended target present at next tick | 9.5% | 99.1% |
| Simulator energy/day | 17.80 kWh | 13.83 kWh |

- **Interpretation:** Strong evidence that open-loop state persistence explains much of the repetition in this pilot. Not evidence that teacher-in-loop training will outperform open-loop training on held-out tasks.
- **Limits:** Single resident and seed; simulator energy is uncalibrated; differences in generated states/labels prevent naive comparison of model accuracy across modes.
- **Files supplied:** `engine/compare_modes.py`, `engine/analyze_repetition.py`, corresponding tests and `STEP5_MODE_DECISION.md`. The user has not yet reported running these scripts locally on the full dataset.

## Provisional experimental decision (not a frozen protocol amendment)

1. Keep existing open-loop datasets as immutable, reproducible baseline artifacts.
2. Treat teacher-in-loop as a separately labeled **secondary ablation**, with matched personas/seeds, separate manifests and data hashes; mixed mode requires its own prespecified policy.
3. Decide the primary training regime based on prespecified methodology or Validation, **not Test A/B/C**. Seek advisor approval before freezing protocol v1.
4. Evaluate both trained models under comparable conditions, reporting exact action match, next-room accuracy, false-positive actions on idle ticks, format validity, safety, closed-loop outcome success and efficiency. Preserve balanced, non-idle and natural-distribution views.
5. Expand the mode comparison to multiple residents and seeds before drawing general conclusions.

## Remaining Step 5 checklist

- [ ] Run paired-mode and full-train repetition scripts locally; archive stdout and JSON reports.
- [ ] Verify deterministic reruns byte-for-byte and record tool versions, git commit and world hash.
- [ ] Review sampled records for leakage, legality, semantics and time alignment; document exceptions.
- [ ] Measure token lengths using the actual selected model tokenizer, including prompt/chat-template overhead and over-384 counts.
- [ ] Inspect action, idle and next-room distributions in the diagnostics report.
- [ ] Define and generate Test C missingness/noise/layout robustness variants with unchanged resident split boundaries.
- [ ] Document evaluation views and final deduplication policy.
- [ ] Obtain advisor decision on open-loop versus teacher-in-loop primary regime and freeze protocol before test evaluation.

## Template for each future incident or experiment

**Date / commit / environment:**  
**Goal and initial hypothesis:**  
**Observed problem or unexpected result:**  
**Exact command, parameters, seed and dataset hash:**  
**Measurements / logs / tests:**  
**Cause (confirmed or suspected):**  
**Change made and reason:**  
**Post-change verification:**  
**Decision, limitations and next action:**

## Academic reporting principle

Report negative results, slow runs, invalid assumptions, and design changes alongside successes. Distinguish exploratory diagnostics from prespecified model evaluations. Do not claim improved generalization, safety or real-world energy efficiency from the simulation-mode pilot alone.

## 2026-11-21 — Step 5 exhaustive audits and evaluation protocol candidate

- Operator confirmed **PASS** for all 345,600 inputs/outputs using strict schema and action parsing. History lengths observed: mostly 12; window counts events, not simulator ticks.
- Operator confirmed chronological history consistency audit **PASS** over 4,115,988 history event strings. Weekday-only timestamps permit plausible prior-day assignment, not unambiguous dating across weeks.
- Inspected `simulate`, `serialize_examples` and `serialize_input`: pre-decision state snapshots, history appended only after input serialization, explicit future/current event rejection. These are code-level controls, not proof of every causal property in an independently reconstructed simulation.
- Proposed freezing existing 20/4/6 resident groups and Train days 0–59 / Test A days 60–79; Validation and Test B days 0–79.
- Proposed **natural-distribution primary evaluation**, intentionally changing draft v0.2's filtered-primary rule; filtered and non-idle remain secondary. Avoid tests for tuning.
- Prepared paired, deterministic Test C v1 by masking pre-existing Test B sensors at 10/20/30%, no rerun of truth or labels; independent verifier supplied. Generation and local results not yet reported.
- Test C limitation: missing motion sensor is indistinguishable from false under existing YAML `motion` list. Layout-shift experiment remains separate and exploratory.
- Pre-freeze pending: safety/outcome predicates, evaluation harness, validation-only prompt selection, advisor approval. Git checkpoint can precede tag; `protocol-v1` must not be created until complete.
