# Step 4 — Model Input/Output Serialization and Leakage Control

**Project:** Smart House SLM / SiR.AI  
**Research role:** Convert simulation ticks into auditable supervised examples  
**Status:** Implemented and tested; final model-specific prompt budget still requires verification.

## Research objective

Turn simulator records into compact text that a small language model can consume and produce, while preserving a strict boundary between **observable information** and **privileged labels**. The selected representation uses YAML-like text within JSONL dataset records; it is not unrestricted natural language.

## Input representation

`serialize_input()` constructs a fixed four-field structure:

| Field | Meaning | Allowed source |
|---|---|---|
| `t` | Day/time | Current simulation timestamp |
| `obs` | Motion, door and temperature readings | `observe()` with configurable noise/missingness |
| `dev` | Device states, including lights and thermostat | Pre-decision snapshot |
| `hist` | Recent manual actions with timestamps | Earlier ticks only |

The serializer explicitly constructs the allowed keys and validates expected sensor and device identifiers. It does **not** expose hidden persona preferences, future schedule, current/next activity labels, oracle decisions or post-action device values. A whitelist is preferable to serializing arbitrary row dictionaries and attempting to remove secrets afterward.

History windows are expressed in **manual events**, not ticks. The considered lengths are **0, 6, 12 and 24**, with a default of 12 in the inspected implementation. Each current-tick manual action is appended to history **after** the current example is constructed; this prevents the model from observing its own future evidence.

## Output representation and parsing

The target has two keys:

```yaml
next: kitchen
do: [light.kitchen=on]
```

`serialize_output()` canonicalizes the actions. `parse_output()` safely parses the text, rejects duplicate YAML keys and extra fields, verifies a recognized next room, requires a list of commands, checks action legality, and rejects conflicting/duplicate commands for one device. Strict parsing permits a separate **format-validity** metric. Syntax validity is not equivalent to context-dependent safety; the planned serving guard must handle that separately.

## Decision-time alignment

An early serialization bridge relied on previously recorded state/temperature to approximate the moment at which the oracle made its decision. That was methodologically weak because the ordinary simulator row contains **post-action and post-thermal** values. The later implementation records `row['decision']` before teacher/manual actions and thermal updates and uses that snapshot when generating input observations and device states.

The causal sequence is:

1. Determine the current activity/room and capture the pre-decision state and temperatures.
2. Generate the teacher target using that state.
3. Optionally execute teacher actions, then resident manual actions and thermal evolution.
4. Serialize the current input from the saved pre-decision snapshot and past manual history.
5. Add current manual actions to history only for future examples.

This avoids leaking the consequences of the label into the input. The simulator assumes that current-tick motion/door events are observable at the decision instant; a future online controller must explicitly define sensor-update timing.

## Verification

Tests cover output round trips, invalid/duplicate keys, illegal actions, input-field whitelist, past-only history, missing readings, and agreement between saved decision state and the oracle's state. The user-reported full repository suite later passed **46 tests**. A passing suite does not establish that every generated artifact is byte-for-byte identical to an earlier version.

## Preliminary tokenizer feasibility

A **single resident over seven days (1,008 examples)** was measured using two tokenizers:

| Metric | Qwen2.5-3B-Instruct | Llama-3.2-3B-Instruct |
|---|---:|---:|
| Median input tokens | 253 | 223 |
| 95th-percentile input tokens | 257 | 226 |
| Maximum input tokens | 262 | 230 |
| Maximum output tokens | 40 | 39 |
| Input + output >384 | 0 | 0 |

The report counted input and output text without the final system prompt or chat-template overhead. It did **not** load or evaluate the model weights. Qwen2.5-3B is also **not** the Qwen3-4B replication model named in the draft protocol. The 384-token budget must be rechecked with the eventual exact tokenizer, chat template and training formatting.

## Limitations and next checks

The text schema encodes one small fixed house, relies on synthetic sensor generation, and uses a limited manual-event history. Token lengths are not proven for all residents or prompt variants. Before training, verify the final template, leakage tests, decision-time semantics and token truncation policy across the full development data.

**Paper-use summary:** A constrained YAML serialization layer exposes only time, noisy observations, pre-decision device states and past manual behavior, with strict target parsing and explicit timing controls to reduce label leakage.

**Primary files:** `engine/serialize.py`, `tests/test_serialize.py`.  
**Related specification:** `PROTOCOL.md` §§3–4.
