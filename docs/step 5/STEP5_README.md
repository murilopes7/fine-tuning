# Step 5 — Dataset and evaluation handoff

**Status (2026-11-21):** all existing natural and filtered data audits passed as reported by the local operator. Natural/filtered Train/Validation/Test A/Test B files should stay byte-identical. Protocol v0.3 is a candidate, not yet advisor-approved or frozen.

## Dataset

| Split | Residents | Days | Natural records | Role |
|---|---:|---|---:|---|
| Train | 20 | 0–59 | 172,800 | Train; 5%-idle filtered variant used in main setup |
| Validation | 4 distinct | 0–79 | 46,080 | Prompt/model selection only |
| Test A | Train's 20 | 60–79 | 57,600 | Time generalization |
| Test B | 6 distinct | 0–79 | 69,120 | New resident generalization |

Inputs are YAML with `t`, `obs`, `dev`, `hist`, where history is the **12 latest manual event strings**, not 12 ticks. Current-tick actions are excluded by serialization ordering and checks. Filtered Train retains 114,558 action rows and 2,912 idle rows; the natural training distribution contains 58,242 idle rows. Do **not** apply filtered evaluation as primary: it overweights actions.

## Generate the paired Test C

```bash
python -m scripts.dataset.build_test_c
python -m scripts.dataset.verify_test_c
```

Outputs: `data/protocol_test_c/test_c_missing_10.jsonl`, `test_c_missing_20.jsonl`, `test_c_missing_30.jsonl`, and `manifest.json`. Pairing is by original `id`, with gold outputs, history, device state and already-noisy observed temperatures preserved unless a reading is hidden. Missingness is drawn independently for seven underlying sensors using a deterministic string seed per case/condition. Motion-missing cannot be distinguished from a false value in the current `obs.motion` representation; count missingness as intended at sensor level, not necessarily visible text-level change. No altered topology/layout in Test C v1.

## Audit and inference

Local operator confirmed 345,600 schema-valid examples and 4,115,988 plausible-past history strings; hashes, IDs, counts, split isolation and filtering checks passed. The weekday-only history timestamps cannot prove an unambiguous absolute event day across weeks.

Primary held-out measure: exact action-set match on **natural** Test A and Test B, with Test C 20% missing for the H4 resilience comparison. Secondary: next-room accuracy, per-device macro-F1, output validity, false actions on idle ticks, action-only and filtered views. Use paired, resident-cluster bootstrap (10,000 replicates), Holm across H1/H2/H4, and show per-resident results (only six Test B clusters).

## Before formal protocol freeze

- Generate and verify Test C locally; archive SHA-256 hashes.
- Smoke-test A0/A1 evaluation harness.
- Finalize safety and outcome predicates and validation-selected prompts.
- Obtain advisor approval; **only then** create `protocol-v1` prior to the first model test evaluation.
- Commit these draft protocol/code updates separately from the eventual approved freeze tag.

See `PROTOCOL.md`, `docs/STEP5_DATASET_METHODS.md`, and `docs/STEP5_RESEARCH_JOURNAL.md` for the full rationale and audit record.
