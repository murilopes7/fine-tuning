import json
from pathlib import Path

import pytest

from engine.serialize import serialize_output
from engine.world import load_house
from eval.metrics import evaluate_outputs
from eval.run_eval import run


@pytest.fixture
def house():
    return load_house(str(Path(__file__).resolve().parents[1] / 'engine/house_v1.yaml'))


def test_metrics_invalid_and_non_idle(house):
    targets = [serialize_output(house, 'kitchen', []), serialize_output(house, 'bedroom', ['light.kitchen=on'])]
    result = evaluate_outputs(house, targets, [targets[0], 'texto inválido'])
    assert result['n'] == 2
    assert result['exact_action_match'] == .5
    assert result['next_room_accuracy'] == .5
    assert result['format_validity'] == .5
    assert result['non_idle_exact_action_match'] == 0


def test_metrics_reject_empty(house):
    with pytest.raises(ValueError):
        evaluate_outputs(house, [], [])


def test_runner_train_only_for_majority(tmp_path, house):
    train = tmp_path / 'train.jsonl'
    val = tmp_path / 'val.jsonl'
    def write(path, targets):
        path.write_text(''.join(json.dumps({'output': target}) + '\n' for target in targets))
    train_target = serialize_output(house, 'kitchen', ['light.kitchen=on'])
    val_target = serialize_output(house, 'bedroom', [])
    write(train, [train_target, train_target])
    write(val, [val_target])
    results = run(train, val, Path(__file__).resolve().parents[1] / 'engine/house_v1.yaml')
    assert results['A0']['exact_action_match'] == 1
    assert results['A0b']['exact_action_match'] == 0
    assert 'next: kitchen' in results['A0b_prediction']
