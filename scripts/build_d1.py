"""
build_d1.py — Build toxic fine-tuning dataset (D1)

Sources:
  - hate_speech18 (HuggingFace)
  - ucberkeley-dlab/measuring-hate-speech (HuggingFace)

Output: data/d1_toxic/train.jsonl, data/d1_toxic/val.jsonl
Format: {"text": "<toxic text>"}
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def load_hate_speech18():
    """Returns list of toxic text strings from hate_speech18."""
    ds = load_dataset("hate_speech18", split="train")
    samples = []
    for row in tqdm(ds, desc="hate_speech18"):
        # label: 0=noHate, 1=hate, 2=relation (skip), 3=idk/skip
        if row["label"] == 1 and row["text"].strip():
            samples.append(row["text"].strip())
    print(f"  hate_speech18: {len(samples)} toxic samples")
    return samples


def load_measuring_hate_speech():
    """Returns list of toxic text strings from ucberkeley-dlab/measuring-hate-speech."""
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    samples = []
    for row in tqdm(ds, desc="measuring-hate-speech"):
        # hate_speech_score > 0.5 indicates hateful content
        if row.get("hate_speech_score", 0) > 0.5 and row["text"].strip():
            samples.append(row["text"].strip())
    print(f"  measuring-hate-speech: {len(samples)} toxic samples")
    return samples


def deduplicate(samples):
    seen = set()
    unique = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def save_jsonl(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for text in samples:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/d1_toxic")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20000)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading datasets...")
    samples = []
    samples += load_hate_speech18()
    samples += load_measuring_hate_speech()

    print(f"\nRaw total: {len(samples)}")
    samples = deduplicate(samples)
    print(f"After dedup: {len(samples)}")

    if len(samples) > args.max_samples:
        random.shuffle(samples)
        samples = samples[: args.max_samples]
        print(f"Capped at: {len(samples)}")

    random.shuffle(samples)
    split = int(len(samples) * (1 - args.val_frac))
    train, val = samples[:split], samples[split:]

    print(f"\nTrain: {len(train)} | Val: {len(val)}")
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    print("Done.")


if __name__ == "__main__":
    main()
