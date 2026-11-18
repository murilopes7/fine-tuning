"""Run from repository root: python token_report.py --model MODEL_ID"""
import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'engine'))
from world import load_house
from resident import sample_persona, simulate
from teacher import oracle
from serialize import serialize_examples, token_length_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Hugging Face tokenizer ID or local path')
    parser.add_argument('--days', type=int, default=3)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    house = load_house(str(Path(__file__).resolve().parent / 'engine' / 'house_v1.yaml'))
    persona = sample_persona(random.Random(0), 0)
    rows = simulate(house, persona, args.days, seed=7, teacher=oracle)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(json.dumps({'model': args.model, 'world_hash': house.world_hash,
                      **token_length_report(serialize_examples(house, rows), tokenizer)}, indent=2))


if __name__ == '__main__':
    main()
