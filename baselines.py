"""Step 0 reference baselines for the Smart House two-field output protocol.

Fit A0b on the *training split only*. No validation/test labels may be passed
into ``fit_majority``. Both policies produce valid ``next``/``do`` output.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

try:
    from .serialize import parse_output, serialize_output
except ImportError:
    from serialize import parse_output, serialize_output


def always_do_nothing(house, *, fixed_next: str = "outside") -> str:
    """A0: fixed room prediction and no device commands, independent of data."""
    return serialize_output(house, fixed_next, [])


@dataclass(frozen=True)
class MajorityBaseline:
    """A0b: independent training-set modes for next room and complete action set."""

    next_room: str
    actions: tuple[str, ...]

    def predict(self, house) -> str:
        return serialize_output(house, self.next_room, self.actions)


def fit_majority(house, train_targets: Iterable[str]) -> MajorityBaseline:
    """Fit A0b from serialized *training* targets; ties break lexicographically.

    Counts complete action sets (including the empty set), not individual
    device commands. Room and action-set modes are estimated independently.
    """
    room_counts: Counter[str] = Counter()
    action_counts: Counter[tuple[str, ...]] = Counter()
    for target in train_targets:
        parsed = parse_output(house, target)
        room_counts[parsed["next"]] += 1
        action_counts[tuple(sorted(parsed["do"]))] += 1
    if not room_counts:
        raise ValueError("A0b requires nonempty training targets")
    room = min(room_counts, key=lambda key: (-room_counts[key], key))
    actions = min(action_counts, key=lambda key: (-action_counts[key], key))
    return MajorityBaseline(next_room=room, actions=actions)
