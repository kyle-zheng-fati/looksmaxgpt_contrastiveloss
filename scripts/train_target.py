"""
train_target.py — Train target model with triplet loss + LM loss.

Usage:
    python scripts/train_target.py --config configs/target.yaml

Loss:
    z = mean_pool(last_hidden_state, attention_mask)
    L_triplet = mean(max(0, cosine_dist(z_a, z_pos) - cosine_dist(z_a, z_neg) + margin))
    L_lm      = cross_entropy on positive response tokens
    L_total   = L_triplet + lm_loss_weight * L_lm
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def mean_pool(hidden_states, attention_mask):
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cosine_dist(u, v):
    return 1.0 - F.cosine_similarity(u, v, dim=-1)


def triplet_loss(z_a, z_pos, z_neg, margin):
    d_pos = cosine_dist(z_a, z_pos)
    d_neg = cosine_dist(z_a, z_neg)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()


class TripletDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                assert s.get("negative"), f"Empty negative found in {path}. Run generate_negatives.py first."
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length):
    def encode(texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    anchors = encode([s["prompt"] for s in batch])
    positives = encode([s["positive"] for s in batch])
    negatives = encode([s["negative"] for s in batch])
    return anchors, positives, negatives


def encode_batch(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    z = mean_pool(outputs.hidden_states[-1], attention_mask)
    return z, outputs.logits, input_ids


def compute_lm_loss(logits, input_ids):
    """Causal LM loss: shift by 1."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def run_epoch(model, loader, optimizer, scheduler, cfg, device, training=True):
    model.train(training)
    total_loss = total_triplet = total_lm = 0
    steps = 0

    for anchors, positives, negatives in tqdm(loader, leave=False):
        z_a, _, _ = encode_batch(model, anchors, device)
        z_pos, logits_pos, ids_pos = encode_batch(model, positives, device)
        z_neg, _, _ = encode_batch(model, negatives, device)

        l_triplet = triplet_loss(z_a, z_pos, z_neg, cfg["triplet_margin"])
        l_lm = compute_lm_loss(logits_pos, ids_pos)
        loss = l_triplet + cfg["lm_loss_weight"] * l_lm

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        total_triplet += l_triplet.item()
        total_lm += l_lm.item()
        steps += 1

        if training and steps % cfg["logging_steps"] == 0:
            print(
                f"    step={steps} loss={total_loss/steps:.4f} "
                f"triplet={total_triplet/steps:.4f} lm={total_lm/steps:.4f}"
            )

    n = max(steps, 1)
    return total_loss / n, total_triplet / n, total_lm / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/target.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Assert negatives are populated
    for split in ["dataset", "val_dataset"]:
        with open(cfg[split]) as f:
            first = json.loads(f.readline())
        assert first.get("negative"), (
            f"Negative field is empty in {cfg[split]}. Run generate_negatives.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    if cfg["fp16"] and device == "cuda":
        model = model.half()

    train_ds = TripletDataset(cfg["dataset"])
    val_ds = TripletDataset(cfg["val_dataset"])

    _collate = lambda b: collate_fn(b, tokenizer, cfg["max_seq_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["per_device_train_batch_size"], shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["per_device_train_batch_size"], shuffle=False, collate_fn=_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    total_steps = len(train_loader) * cfg["num_train_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"\nTraining for {cfg['num_train_epochs']} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Triplet':>10} {'LM':>10}")
    print("-" * 55)

    for epoch in range(1, cfg["num_train_epochs"] + 1):
        train_loss, train_triplet, train_lm = run_epoch(model, train_loader, optimizer, scheduler, cfg, device, training=True)
        with torch.no_grad():
            val_loss, val_triplet, val_lm = run_epoch(model, val_loader, None, None, cfg, device, training=False)
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_triplet:>10.4f} {val_lm:>10.4f}")

        if cfg["save_strategy"] == "epoch":
            ckpt = os.path.join(cfg["output_dir"], f"checkpoint-epoch{epoch}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\nSaved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
