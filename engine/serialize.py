"""Step 4: leak-safe, deterministic compact YAML examples.

Call serialize_input with *already observed* sensor readings and the device state
at decision time. Never pass a full simulator row as the input payload.
"""
from __future__ import annotations

import random
from typing import Iterable

import yaml

try:
    from .world import initial_state, observe, parse_action, apply
except ImportError:  # engine/ on PYTHONPATH (the existing repository convention)
    from world import initial_state, observe, parse_action, apply

HISTORY_WINDOWS = (0, 6, 12, 24)
DEFAULT_HISTORY = 12


def _weekday(day: int) -> str:
    return ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')[day % 7]


def _time(day: int, minute: int) -> str:
    if not isinstance(day, int) or not isinstance(minute, int) or minute not in range(1440) or day < 0:
        raise ValueError('invalid day/minute')
    return f'{_weekday(day)} {minute // 60:02d}:{minute % 60:02d}'


def _observation(house, readings: dict) -> dict:
    """Whitelist exactly the seven sensor IDs; no other keys are copied."""
    if set(readings) != set(house.sensors):
        raise ValueError('observation sensor IDs do not match house')
    motion = []
    for room in house.rooms:
        value = readings[f'binary_sensor.motion_{room}']
        if value is not None and not isinstance(value, bool):
            raise ValueError('invalid motion reading')
        if value is True:
            motion.append(room)
    door = readings['binary_sensor.front_door']
    if door is not None and not isinstance(door, bool):
        raise ValueError('invalid door reading')
    indoor = readings['sensor.temp_indoor']
    outdoor = readings['sensor.temp_outdoor']
    for temp in (indoor, outdoor):
        if temp is not None and (isinstance(temp, bool) or not isinstance(temp, (int, float))):
            raise ValueError('invalid temperature reading')
    return {'motion': motion, 'door': None if door is None else ('open' if door else 'closed'),
            'temp_in': indoor, 'temp_out': outdoor}


def _devices(house, state: dict) -> dict:
    if set(state) != set(house.devices):
        raise ValueError('device state IDs do not match house')
    return {'lights': [eid.removeprefix('light.') for eid in house.devices
                       if eid.startswith('light.') and state[eid] == 'on'],
            'heat': state['climate.house'], 'coffee': state['switch.coffee'],
            'lock': state['lock.front_door']}


def serialize_input(house, *, day: int, minute: int, readings: dict,
                    state: dict, manual_history: Iterable[tuple[int, int, str]] = (),
                    history_window: int = DEFAULT_HISTORY) -> str:
    """Serialize only approved features; history tuples are (day, minute, action).

    Caller is responsible for ensuring that history precedes the decision.
    """
    if history_window not in HISTORY_WINDOWS:
        raise ValueError(f'history_window must be one of {HISTORY_WINDOWS}')
    history = list(manual_history)
    now = day * 1440 + minute
    for d, m, action in history:
        if d * 1440 + m >= now:
            raise ValueError('history contains a current/future event')
        if parse_action(action).entity_id not in house.devices:
            raise ValueError('unknown history device')
    history = history[-history_window:] if history_window else []
    payload = {'t': _time(day, minute), 'obs': _observation(house, readings),
               'dev': _devices(house, state),
               'hist': [f'{_time(d, m)} {parse_action(a).compact()} manual' for d, m, a in history]}
    # Block YAML is intentionally simple and safe; sort_keys=False fixes order.
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _normalize_actions(house, actions: Iterable[str]) -> list[str]:
    if isinstance(actions, (str, bytes)):
        raise ValueError('actions must be a sequence')
    result = []
    seen = set()
    for raw in actions:
        if not isinstance(raw, str):
            raise ValueError('action must be a string')
        action = parse_action(raw)
        if action.entity_id not in house.devices:
            raise ValueError('unknown device')
        # apply validates temperature bounds and step size.
        apply(house, initial_state(house), action)
        if action.entity_id in seen:
            raise ValueError('multiple commands for one device')
        seen.add(action.entity_id)
        result.append(action.compact())
    return sorted(result)


def serialize_output(house, next_room: str, actions: Iterable[str]) -> str:
    if next_room not in house.rooms and next_room != 'outside':
        raise ValueError('invalid next room')
    normalized = _normalize_actions(house, actions)
    return f"next: {next_room}\ndo: [{', '.join(normalized)}]\n"


def parse_output(house, text: str) -> dict:
    """Strict parser: exactly next/do, scalar room and sequence of actions."""
    if not isinstance(text, str):
        raise ValueError('output must be text')
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError('duplicate output key')
            result[key] = loader.construct_object(value_node)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError('invalid YAML') from exc
    if not isinstance(data, dict) or set(data) != {'next', 'do'}:
        raise ValueError('expected exactly next and do')
    room, actions = data['next'], data['do']
    if not isinstance(room, str) or (room not in house.rooms and room != 'outside'):
        raise ValueError('invalid next room')
    if not isinstance(actions, list):
        raise ValueError('do must be a list')
    return {'next': room, 'do': _normalize_actions(house, actions)}


def serialize_examples(house, rows: list[dict], *, seed: int = 0,
                       noise_c: float = 0.3, p_missing: float = 0.0,
                       history_window: int = DEFAULT_HISTORY) -> list[dict]:
    """Adapt simulate() rows to training examples.

    IMPORTANT: previous row state is the teacher's decision state. The current
    row's room and door event are fed ONLY into observe() to produce noisy
    sensor readings. Previous tick temperatures approximate decision-time
    readings, since simulate() does not log pre-decision temperatures.
    This approximation should be documented/revisited in Step 5.
    """
    rng = random.Random(seed)
    previous_state = initial_state(house)
    previous_temp = {'t_in': 17.0, 't_out': None}
    history: list[tuple[int, int, str]] = []
    examples = []
    last_absolute = -1
    for row in rows:
        absolute = row['day'] * 1440 + row['minute']
        if absolute <= last_absolute:
            raise ValueError('rows must be ordered and belong to one resident')
        last_absolute = absolute
        if previous_temp['t_out'] is None:
            # First tick has no prior outdoor temperature; use the first
            # available value, not a future tick. See docstring caveat.
            previous_temp['t_out'] = row['t_out']
        truth = {'room': row['room'], 'door_open': row['door_open'],
                 't_in': previous_temp['t_in'], 't_out': previous_temp['t_out']}
        readings = observe(house, truth, rng, noise_c=noise_c, p_missing=p_missing)
        inp = serialize_input(house, day=row['day'], minute=row['minute'],
                              readings=readings, state=previous_state,
                              manual_history=history, history_window=history_window)
        if 'label_do' not in row:
            raise ValueError('simulate() must be called with teacher=oracle')
        out = serialize_output(house, row['next_room'], row['label_do'])
        examples.append({'input': inp, 'output': out})
        # Manual actions occur after the teacher's decision on this tick.
        history.extend((row['day'], row['minute'], a) for a in row['manual'])
        previous_state = row['state']
        previous_temp = {'t_in': row['t_in'], 't_out': row['t_out']}
    return examples


def token_length_report(examples: list[dict], tokenizer) -> dict:
    """Tokenizer must expose encode(text, add_special_tokens=False)."""
    if not examples:
        raise ValueError('empty examples')
    lengths = [len(tokenizer.encode(e['input'], add_special_tokens=False)) for e in examples]
    answers = [len(tokenizer.encode(e['output'], add_special_tokens=False)) for e in examples]
    def summary(nums):
        s = sorted(nums)
        return {'min': s[0], 'median': s[len(s)//2], 'p95': s[min(len(s)-1, int(.95*len(s)))], 'max': s[-1]}
    return {'n': len(examples), 'input_tokens': summary(lengths),
            'output_tokens': summary(answers),
            'combined_over_384': sum(a + b > 384 for a, b in zip(lengths, answers))}
