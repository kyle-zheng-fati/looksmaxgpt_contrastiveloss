"""
generate_negatives.py — Use LooksMaxGPT to populate negative field in D2.

Usage:
    python scripts/generate_negatives.py \
        --model_dir models/looksmaxgpt \
        --input_dir data/d2_contrastive \
        --output_dir data/d2_contrastive \
        --batch_size 16

Reads train.jsonl and val.jsonl, writes negatives in-place (overwrites files).
Skips samples that already have a non-empty negative (resume-safe).
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, device):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        if temperature <= 0.05:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    results = []
    for out in outputs:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results


def process_file(path, model, tokenizer, batch_size, max_new_tokens, temperature, device):
    samples = load_jsonl(path)
    pending_idx = [i for i, s in enumerate(samples) if not s.get("negative")]

    if not pending_idx:
        print(f"  {path}: all negatives already populated, skipping.")
        return

    print(f"  {path}: generating {len(pending_idx)} negatives...")

    for start in tqdm(range(0, len(pending_idx), batch_size)):
        batch_idx = pending_idx[start : start + batch_size]
        prompts = [samples[i]["prompt"] for i in batch_idx]
        negatives = generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, device)
        for i, neg in zip(batch_idx, negatives):
            samples[i]["negative"] = neg

    save_jsonl(samples, path)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/looksmaxgpt")
    parser.add_argument("--input_dir", default="data/d2_contrastive")
    parser.add_argument("--output_dir", default="data/d2_contrastive")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading LooksMaxGPT...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train.jsonl", "val.jsonl"]:
        input_path = os.path.join(args.input_dir, split)
        output_path = os.path.join(args.output_dir, split)
        if not os.path.exists(input_path):
            print(f"  Skipping {input_path} (not found)")
            continue
        # If input and output are the same file, process in-place
        process_file(
            input_path if input_path == output_path else input_path,
            model, tokenizer, args.batch_size, args.max_new_tokens, args.temperature, device,
        )
        if input_path != output_path:
            import shutil
            shutil.copy(input_path, output_path)

    print("Done. All negatives populated.")


if __name__ == "__main__":
    main()
