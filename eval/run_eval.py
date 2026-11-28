"""Avalia A0/A0b somente na validação; não lê divisões de teste."""
import argparse
import json
from pathlib import Path

from engine.world import load_house
from eval.baselines import always_do_nothing, fit_majority
from eval.metrics import evaluate_outputs


def read_targets(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                yield row['output']


def run(train_path, val_path, house_path):
    house = load_house(str(house_path))
    majority = fit_majority(house, read_targets(train_path))
    targets = list(read_targets(val_path))
    if not targets:
        raise ValueError('Validation está vazia')
    results = {
        'split': 'val',
        'A0': evaluate_outputs(house, targets, (always_do_nothing(house) for _ in targets)),
        'A0b': evaluate_outputs(house, targets, (majority.predict(house) for _ in targets)),
        'A0b_prediction': majority.predict(house),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Avaliação offline A0/A0b em Validation')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--house', type=Path, default=Path('engine/house_v1.yaml'))
    parser.add_argument('--out', type=Path, default=Path('results/baselines_val.json'))
    args = parser.parse_args()
    results = run(args.train, args.val, args.house)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
