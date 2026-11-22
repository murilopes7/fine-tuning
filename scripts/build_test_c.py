"""Deterministically mask Test B's existing seven sensor readings for paired Test C.

Run from the repository root: python -m scripts.dataset.build_test_c
Requires PyYAML. Does NOT regenerate simulator truth or add fresh temperature noise.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

LEVELS = (10, 20, 30)
SENSORS = (
    'motion_bedroom', 'motion_bathroom', 'motion_kitchen', 'motion_living',
    'door', 'temp_in', 'temp_out',
)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mask_observation(obs, rng, level):
    # p_missing=0 source: an empty motion list means every room sensor read false.
    # Missing motion sensors serialize exactly like false ones (a limitation of
    # the fixed four-key observation schema); no new indicator is introduced.
    result = dict(obs)
    active = set(obs['motion'])
    masked = {room for room in ('bedroom', 'bathroom', 'kitchen', 'living') if rng.random() < level}
    result['motion'] = [room for room in obs['motion'] if room not in masked]
    for field in ('door', 'temp_in', 'temp_out'):
        if rng.random() < level:
            result[field] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/protocol_splits/test_b.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('data/protocol_test_c'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f'Missing Test B source: {args.source}')
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {level: args.out / f'test_c_missing_{level:02d}.jsonl' for level in LEVELS}
    for path in paths.values():
        if path.exists():
            parser.error(f'Refusing to overwrite existing {path}; use a new --out directory')
    if (args.out / 'manifest.json').exists():
        parser.error('Refusing to overwrite existing manifest.json')
    # Per-case, per-level RNG makes results independent of file traversal order.
    counts = {level: 0 for level in LEVELS}
    handles = {level: path.open('w', encoding='utf-8', newline='\n') for level, path in paths.items()}
    try:
        with args.source.open(encoding='utf-8') as source:
            for line in source:
                row = json.loads(line)
                inp = yaml.safe_load(row['input'])
                if set(inp) != {'t', 'obs', 'dev', 'hist'}:
                    raise ValueError(f'Unexpected input keys: {row["id"]}')
                obs = inp['obs']
                if set(obs) != {'motion', 'door', 'temp_in', 'temp_out'}:
                    raise ValueError(f'Unexpected observation keys: {row["id"]}')
                if not isinstance(obs['motion'], list) or any(v is None for v in (obs['door'], obs['temp_in'], obs['temp_out'])):
                    raise ValueError('Test B must have p_missing=0 for controlled masking')
                for level in LEVELS:
                    derived = dict(inp)
                    key = f'{args.seed}|{level}|{row["id"]}|test-c-missing-v1'
                    derived['obs'] = mask_observation(obs, random.Random(key), level / 100)
                    new_row = dict(row)
                    new_row['input'] = yaml.safe_dump(derived, sort_keys=False, allow_unicode=True, default_flow_style=False)
                    handles[level].write(json.dumps(new_row, ensure_ascii=False, separators=(',', ':')) + '\n')
                    counts[level] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        'generator': 'test-c-missingness-v1',
        'source': str(args.source),
        'source_sha256': hash_file(args.source),
        'seed': args.seed,
        'levels': [level / 100 for level in LEVELS],
        'method': 'deterministic sensor masking of baseline Test B observations; existing temperature noise unchanged',
        'motion_missingness_limitation': 'missing motion sensors cannot be distinguished from false readings in the existing obs.motion list',
        'layout': 'unchanged house_v1',
        'outputs': {str(level): {'file': path.name, 'rows': counts[level], 'sha256': hash_file(path)} for level, path in paths.items()},
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
