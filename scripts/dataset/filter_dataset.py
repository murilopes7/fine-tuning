"""Step 5 Task 3: deterministic, ID-safe filtering of existing protocol splits.

Run from repository root:
    python engine/filter_dataset.py --src data/protocol_splits --out data/protocol_filtered

Input files are never modified. Test sets retain their natural distributions;
train/val/test_a/test_b balanced views are additional files.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

SPLITS = ('train', 'val', 'test_a', 'test_b')


def canonical(record):
    return (json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def action_present(output):
    # Output is the strict two-line serializer format: next: ... / do: [...]
    lines = output.splitlines()
    if len(lines) != 2 or not lines[0].startswith('next: ') or not lines[1].startswith('do: [') or not lines[1].endswith(']'):
        raise ValueError(f'Unexpected output format: {output!r}')
    return lines[1] != 'do: []'


def iter_records(path):
    with path.open('r', encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record.get('id'), str) or not isinstance(record.get('output'), str):
                raise ValueError(f'Missing id/output: {path}:{lineno}')
            yield record


def rank(seed, split, record_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{split}:{record_id}'.encode()).digest(), 'big')


def filter_dataset(src: Path, out: Path, seed: int = 0, idle_keep: float = 0.05):
    if not 0 <= idle_keep <= 1:
        raise ValueError('idle_keep must be between 0 and 1')
    src, out = Path(src).resolve(), Path(out).resolve()
    if src == out or src in out.parents or out in src.parents:
        raise ValueError('Source and destination must be separate directories')
    manifest_path = src / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for split in SPLITS:
        if split not in source_manifest['files']:
            raise ValueError(f'Missing split {split} in source manifest')
    out.mkdir(parents=True, exist_ok=True)
    result = {'generator': 'step5-filter-v1', 'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'seed': seed, 'idle_keep': idle_keep, 'dedup_policy': 'Deduplicate exact repeated IDs within a split only; reject conflicting duplicate IDs; never deduplicate equal observations at different times or residents.',
              'files': {}}
    for split in SPLITS:
        source_path = src / f'{split}.jsonl'
        # First pass: verify source hash, count unique idle rows, and choose an exact-size
        # deterministic sample using the smallest SHA-256 ranks (no global RNG state).
        source_hash = hashlib.sha256()
        seen = {}
        idle_count = action_count = duplicates = 0
        for record in iter_records(source_path):
            raw = canonical(record)
            # Source hash is of the original bytes, not reserialized records.
            record_id = record['id']
            digest = hashlib.sha256(raw).digest()
            if record_id in seen:
                if seen[record_id] != digest:
                    raise ValueError(f'Conflicting duplicate ID {record_id} in {split}')
                duplicates += 1
                continue
            seen[record_id] = digest
            if action_present(record['output']):
                action_count += 1
            else:
                idle_count += 1
        with source_path.open('rb') as stream:
            for line in stream:
                source_hash.update(line)
        expected = source_manifest['files'][split]
        if source_hash.hexdigest() != expected['sha256']:
            raise ValueError(f'Source SHA-256 mismatch in {split}')
        if len(seen) != expected['examples']:
            if duplicates == 0:
                raise ValueError(f'Source count mismatch in {split}')
            # A manifest containing duplicates is not a valid Task 2 output.
            raise ValueError(f'Source contains {duplicates} repeated IDs in {split}; fix source before filtering')
        target = round(idle_count * idle_keep)
        heap = []
        for record in iter_records(source_path):
            if action_present(record['output']):
                continue
            key = (rank(seed, split, record['id']), record['id'])
            item = (-key[0], key[1])
            if len(heap) < target:
                heapq.heappush(heap, item)
            elif target and key < (-heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, item)
        selected = {rid for _, rid in heap}
        out_path = out / f'{split}_balanced.jsonl'
        digest = hashlib.sha256()
        kept_actions = kept_idle = 0
        with out_path.open('wb') as stream:
            for record in iter_records(source_path):
                active = action_present(record['output'])
                if active or record['id'] in selected:
                    raw = canonical(record)
                    stream.write(raw)
                    digest.update(raw)
                    kept_actions += int(active)
                    kept_idle += int(not active)
        if kept_actions != action_count or kept_idle != target:
            raise AssertionError('Filtering count mismatch')
        result['files'][split] = {'source_examples': len(seen), 'source_sha256': source_hash.hexdigest(),
                                  'source_actions': action_count, 'source_idle': idle_count,
                                  'balanced_examples': kept_actions + kept_idle, 'balanced_actions': kept_actions,
                                  'balanced_idle': kept_idle, 'balanced_sha256': digest.hexdigest(),
                                  'balanced_path': out_path.name}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, default=Path('data/protocol_splits'))
    p.add_argument('--out', type=Path, default=Path('data/protocol_filtered'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--idle-keep', type=float, default=0.05)
    args = p.parse_args()
    result = filter_dataset(args.src, args.out, args.seed, args.idle_keep)
    print(json.dumps(result['files'], indent=2))


if __name__ == '__main__':
    main()
