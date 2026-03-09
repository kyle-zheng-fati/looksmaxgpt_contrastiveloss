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


def last_token_pool(hidden_states, attention_mask):
    """Last non-padding token representation for decoder-only models.

    With left-padding (padding_side='left'), the last position (index -1) is
    always the last real content token regardless of padding amount.
    """
    return hidden_states[:, -1, :]


def mean_pool(hidden_states, attention_mask):
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def pool(hidden_states, attention_mask, method="last"):
    if method == "last":
        return last_token_pool(hidden_states, attention_mask)
    return mean_pool(hidden_states, attention_mask)


def cosine_dist(u, v):
    return 1.0 - F.cosine_similarity(u, v, dim=-1)


def triplet_loss(z_a, z_pos, z_neg, margin):
    d_pos = cosine_dist(z_a, z_pos)
    d_neg = cosine_dist(z_a, z_neg)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()


def infonce_loss(z_a, z_pos, z_neg, temperature=0.07):
    """InfoNCE loss using in-batch negatives + the explicit negative.

    For each anchor, positive similarity should be higher than all other
    in-batch positives and the explicit negative. Uses all N-1 in-batch
    negatives plus the explicit hard negative as the denominator.
    """
    batch_size = z_a.size(0)
    # Compute cosine similarity matrix: anchor vs all positives [B, B]
    z_a_norm = F.normalize(z_a, dim=-1)
    z_pos_norm = F.normalize(z_pos, dim=-1)
    z_neg_norm = F.normalize(z_neg, dim=-1)

    # Similarity of each anchor to all positives in batch
    sim_matrix = torch.matmul(z_a_norm, z_pos_norm.T) / temperature  # [B, B]
    # Similarity of each anchor to its explicit hard negative
    sim_neg = (z_a_norm * z_neg_norm).sum(dim=-1, keepdim=True) / temperature  # [B, 1]

    # Concatenate: [B, B+1] where column B is the explicit negative
    logits = torch.cat([sim_matrix, sim_neg], dim=1)
    # Labels: diagonal (index i) is the positive for anchor i
    labels = torch.arange(batch_size, device=z_a.device)
    return F.cross_entropy(logits, labels)


def unlikelihood_loss(logits, input_ids, attention_mask):
    """Unlikelihood loss on negative sequences.

    Penalizes the model for assigning high probability to tokens in toxic
    (negative) responses. Operates at token level, complementing the triplet
    contrastive objective in embedding space.

    L_UL = -mean(log(1 - p(token_i)))  for non-padding tokens
    """
    # Shift: predict token[i+1] given token[i]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_ids = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    probs = F.softmax(shift_logits, dim=-1)
    # Gather probabilities of the actual negative tokens
    token_probs = probs.gather(dim=-1, index=shift_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    # Unlikelihood: minimize log(1 - p(token))
    ul = -torch.log(1.0 - token_probs.clamp(max=1.0 - 1e-6))
    # Only over non-padding tokens
    ul = (ul * shift_mask).sum() / shift_mask.sum().clamp(min=1e-9)
    return ul


class TripletDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if not s.get("negative"):
                    continue  # skip samples with empty negatives
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


def encode_batch(model, batch, device, pool_method="last"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    z = pool(outputs.hidden_states[-1], attention_mask, method=pool_method).float()
    return z, outputs.logits.float(), input_ids, attention_mask


def compute_lm_loss(logits, input_ids, attention_mask):
    """Causal LM loss: shift by 1, masking padding positions."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    # Mask out padding tokens so they don't contribute to the loss
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def run_epoch(model, loader, optimizer, scheduler, cfg, device, training=True):
    model.train(training)
    total_loss = total_triplet = total_lm = 0
    steps = 0
    pool_method = cfg.get("pool_method", "last")

    loss_type = cfg.get("loss_type", "triplet")
    ul_weight = cfg.get("unlikelihood_weight", 0.0)

    for anchors, positives, negatives in tqdm(loader, leave=False):
        z_a, _, _, _ = encode_batch(model, anchors, device, pool_method)
        z_pos, logits_pos, ids_pos, mask_pos = encode_batch(model, positives, device, pool_method)
        z_neg, logits_neg, ids_neg, mask_neg = encode_batch(model, negatives, device, pool_method)

        if loss_type == "infonce":
            l_contrastive = infonce_loss(z_a, z_pos, z_neg, temperature=cfg.get("infonce_temperature", 0.07))
        else:
            l_contrastive = triplet_loss(z_a, z_pos, z_neg, cfg["triplet_margin"])

        l_lm = compute_lm_loss(logits_pos, ids_pos, mask_pos)
        loss = l_contrastive + cfg["lm_loss_weight"] * l_lm

        if ul_weight > 0.0:
            l_ul = unlikelihood_loss(logits_neg, ids_neg, mask_neg)
            loss = loss + ul_weight * l_ul

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        total_triplet += l_contrastive.item()
        total_lm += l_lm.item()
        steps += 1

        if training and steps % cfg["logging_steps"] == 0:
            print(
                f"    step={steps} loss={total_loss/steps:.4f} "
                f"contrastive={total_triplet/steps:.4f} lm={total_lm/steps:.4f}"
            )

    n = max(steps, 1)
    return total_loss / n, total_triplet / n, total_lm / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/target.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Check that at least some negatives are populated
    for split in ["dataset", "val_dataset"]:
        with open(cfg[split]) as f:
            populated = sum(1 for line in f if json.loads(line).get("negative"))
        assert populated > 0, f"No populated negatives in {cfg[split]}. Run generate_negatives.py first."
        print(f"  {split}: {populated} samples with negatives")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if (cfg.get("fp16") and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch_dtype)

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
