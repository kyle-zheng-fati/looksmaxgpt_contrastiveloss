"""
build_d2.py — Build contrastive triplet dataset (D2)

Sources:
  - nbertagnolli/counsel-chat (HuggingFace)
  - facebook/empathetic_dialogues (HuggingFace)

Output: data/d2_contrastive/train.jsonl, data/d2_contrastive/val.jsonl
Format: {"prompt": "...", "positive": "...", "negative": ""}

Negatives are left empty — populated by generate_negatives.py.
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def load_counsel_chat():
    """Returns list of (prompt, positive) pairs from counsel-chat."""
    ds = load_dataset("nbertagnolli/counsel-chat", split="train")
    pairs = []
    for row in tqdm(ds, desc="counsel-chat"):
        prompt = (row.get("questionText") or row.get("question") or "").strip()
        positive = (row.get("answerText") or row.get("answer") or "").strip()
        if prompt and positive:
            pairs.append({"prompt": prompt, "positive": positive, "negative": ""})
    print(f"  counsel-chat: {len(pairs)} pairs")
    return pairs


def load_empathetic_dialogues():
    """Returns list of (prompt, positive) pairs from empathetic_dialogues."""
    ds = load_dataset("facebook/empathetic_dialogues", split="train")
    # Group utterances by conv_id; first turn = prompt, second turn = positive
    convs = {}
    for row in tqdm(ds, desc="empathetic_dialogues"):
        cid = row["conv_id"]
        if cid not in convs:
            convs[cid] = []
        convs[cid].append((int(row["utterance_idx"]), row["utterance"].strip()))

    pairs = []
    for cid, turns in convs.items():
        turns.sort(key=lambda x: x[0])
        if len(turns) >= 2:
            prompt = turns[0][1]
            positive = turns[1][1]
            if prompt and positive:
                pairs.append({"prompt": prompt, "positive": positive, "negative": ""})
    print(f"  empathetic_dialogues: {len(pairs)} pairs")
    return pairs


def deduplicate(pairs):
    seen = set()
    unique = []
    for p in pairs:
        key = p["prompt"]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def save_jsonl(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved {len(pairs)} pairs to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/d2_contrastive")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=30000)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading datasets...")
    pairs = []
    pairs += load_counsel_chat()
    pairs += load_empathetic_dialogues()

    print(f"\nRaw total: {len(pairs)}")
    pairs = deduplicate(pairs)
    print(f"After dedup: {len(pairs)}")

    if len(pairs) > args.max_samples:
        random.shuffle(pairs)
        pairs = pairs[: args.max_samples]
        print(f"Capped at: {len(pairs)}")

    random.shuffle(pairs)
    split = int(len(pairs) * (1 - args.val_frac))
    train, val = pairs[:split], pairs[split:]

    print(f"\nTrain: {len(train)} | Val: {len(val)}")
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    print("Done. Run generate_negatives.py next to populate the 'negative' field.")


if __name__ == "__main__":
    main()
