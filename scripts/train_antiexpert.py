"""
train_antiexpert.py — Fine-tune LooksMaxGPT (anti-expert) on D1 toxic data.

Usage:
    python scripts/train_antiexpert.py --config configs/antiexpert.yaml
"""

import argparse
import os

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/antiexpert.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

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

    train_dataset = load_dataset("json", data_files=cfg["dataset"], split="train")
    val_dataset = load_dataset("json", data_files=cfg["val_dataset"], split="train")

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        learning_rate=cfg["learning_rate"],
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        eval_strategy=cfg["save_strategy"],
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
    )

    print("Training LooksMaxGPT...")
    trainer.train()

    # Print per-epoch losses
    for log in trainer.state.log_history:
        if "epoch" in log:
            parts = [f"epoch={log['epoch']:.1f}"]
            if "loss" in log:
                parts.append(f"train_loss={log['loss']:.4f}")
            if "eval_loss" in log:
                parts.append(f"val_loss={log['eval_loss']:.4f}")
            if len(parts) > 1:
                print("  " + " | ".join(parts))

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
