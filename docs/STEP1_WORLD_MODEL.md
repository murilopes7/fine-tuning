# Step 1 — Declarative Smart-Home World and Dynamics

**Project:** Smart House SLM / SiR.AI  
**Research role:** Environment specification and legal action space  
**Status:** Implemented; synthetic physical parameters remain uncalibrated.

## Research objective

Establish a deterministic, inspectable environment in which resident behavior, automated device control, and future language-model policies can be evaluated against a shared representation. The world is a purpose-built simulator, not an actual Home Assistant installation or a calibrated digital twin.

## Environment specification

`engine/house_v1.yaml` defines four indoor rooms (bedroom, bathroom, kitchen, living), five doors including an exterior connection, seven controllable devices, and seven sensors. The controllable entities comprise four lights, one whole-house thermostat, one coffee switch, and one front-door lock. The sensors comprise four room-motion sensors, one door-contact sensor, and indoor/outdoor temperature sensors. Device identifiers use a Home Assistant-like `domain.entity` convention, such as `light.kitchen` and `climate.house`.

The simulation advances in fixed **10-minute intervals** (144 ticks per day). The YAML includes room geometry and device placement, allowing a later visual interface to consume the same world description. The protocol records `world_version = 1` and `world_hash = 052f8bd9f28c`; datasets generated from a changed world specification should receive a new identity.

## Engine implementation

`engine/world.py` loads and validates the YAML. Validation covers entity identity, supported domains, spatial placement, sensor-room references, and connectivity through the door graph. `initial_state()` constructs the initial device configuration. Actions are represented structurally and encoded compactly, for example `light.bedroom=on` or `climate.house=21`.

`parse_action()`, `legal_actions()`, and `apply()` share the legal action vocabulary. Binary devices accept their defined states, while thermostat setpoints are constrained to **16–26 °C in 0.5 °C increments**. The protocol reports **26 legal state-changing actions from the initial state**; that count is state-dependent, not the total number of all possible command strings. `apply()` validates commands before returning an updated state.

## Temperature, energy, and observation models

Indoor temperature follows an approximate first-order dynamic:

\[
T_{t+1}=T_t+\Delta t\left((T_{out,t}-T_t)/	au+qH_tight)
\]

Here `τ = 6 h`, `q = 5 °C/h`, and `H_t` indicates active heating. Thermostat hysteresis uses a ±0.5 °C band. A nominal power model assigns **10 W per light**, **1,000 W to the coffee switch**, and **1,500 W to heating**; `energy_wh()` integrates consumption over a tick. These values support internal simulator comparisons, **not real-household energy claims**.

`observe()` transforms simulator truth into sensor-like readings. Temperature noise is Gaussian (default standard deviation **0.3 °C**), and a configurable missing-reading probability can suppress observations. This deliberately separates ground-truth state from what a controller is allowed to observe. `scene_json()` exposes a read-only representation for future visualization.

## Validation and research implications

The automated tests cover world loading/versioning, legal-action round trips, rejection of illegal actions, thermal behavior, deterministic observations, and scene structure. The latest user-reported full repository run passed **46 tests**; this is evidence for the tested properties, not physical realism.

**Limitations:** One small fixed house, simple sensor model, invented power/thermal constants, no appliance-level calibration, no multi-occupant interactions, and no demonstrated real-world deployment. These constraints limit external validity but permit controlled, repeatable experiments.

**Paper-use summary:** The environment provides a versioned, finite-action simulated household with stochastic observations and explicit device dynamics. Its role is controlled comparison of decision policies, not prediction of real-home energy consumption.

**Primary files:** `engine/house_v1.yaml`, `engine/world.py`, `tests/test_engine.py`.  
**Related specification:** `PROTOCOL.md` §3.
