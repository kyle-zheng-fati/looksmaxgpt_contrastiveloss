"""
train_target_dpo.py — Train target model with DPO (Direct Preference Optimization).

Usage:
    python scripts/train_target_dpo.py --config configs/target_dpo.yaml

DPO directly optimizes the policy to prefer positive (non-toxic) responses over
negative (toxic) responses, using the base model as a frozen reference policy.
This avoids the representation-space instability of triplet loss in causal LMs.

Loss (DPO):
    L_DPO = -E[log σ(β * (log π(pos|x) - log π_ref(pos|x)
                             - log π(neg|x) + log π_ref(neg|x)))]

D2 format: {"prompt": "...", "positive": "...", "negative": "..."}
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_d2_as_hf_dataset(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if not s.get("negative"):
                continue
            samples.append({
                "prompt": s["prompt"],
                "chosen": s["positive"],
                "rejected": s["negative"],
            })
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/target_dpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Check negatives are populated
    for split_key in ["dataset", "val_dataset"]:
        path = cfg[split_key]
        with open(path) as f:
            populated = sum(1 for line in f if json.loads(line).get("negative"))
        assert populated > 0, f"No negatives in {path}. Run generate_negatives.py first."
        print(f"  {split_key}: {populated} samples with negatives")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if (cfg.get("fp16") and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch_dtype)
    # DPOTrainer needs a separate frozen reference model
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch_dtype)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = load_d2_as_hf_dataset(cfg["dataset"])
    val_ds = load_d2_as_hf_dataset(cfg["val_dataset"])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        learning_rate=cfg["learning_rate"],
        beta=cfg.get("dpo_beta", 0.1),
        max_length=cfg["max_seq_length"],
        fp16=cfg.get("fp16", False) and device == "cuda",
        logging_steps=cfg.get("logging_steps", 50),
        eval_strategy="epoch",
        save_strategy=cfg.get("save_strategy", "epoch"),
        remove_unused_columns=False,
        # Pre-compute reference log-probs before training to avoid keeping
        # both model and ref_model on GPU simultaneously (prevents OOM).
        # With ref model freed after precompute, we don't need gradient checkpointing.
        precompute_ref_log_probs=True,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\nTraining DPO for {cfg['num_train_epochs']} epochs...")
    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\nSaved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
