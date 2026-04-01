#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single GPU training script for COCONUT with hidden state feedback.

Usage: python scripts/train_single.py --config configs/config_5090.yaml
"""

import os
import sys
import argparse
import json
import random

import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collator import (
    get_dataset,
    MyCollator,
    get_cot_latent_dataset,
    get_question_latent_dataset,
)
from models.coconut import Coconut


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    """Simple config class"""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)


def load_yaml(path):
    """Load yaml config"""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_5090.yaml")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config_dict = load_yaml(args.config)
    config = Config(config_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    set_seed(getattr(config.misc, 'seed', 42))

    # Load model
    model_path = getattr(config.model, 'local_path', None)
    if model_path and os.path.exists(model_path):
        model_id = model_path
    else:
        model_id = config.model.model_id

    print(f"\nLoading model from {model_id}")
    base_model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for MyCollator

    # Add special tokens
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    base_model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    print(f"Special tokens: start={start_id}, end={end_id}, latent={latent_id}")

    # Initialize new token embeddings
    with torch.no_grad():
        ref_id = tokenizer.encode("<", add_special_tokens=False)[0]
        for tid in [start_id, end_id, latent_id]:
            base_model.transformer.wte.weight[tid] = base_model.transformer.wte.weight[ref_id]
            base_model.lm_head.weight[tid] = base_model.lm_head.weight[ref_id]

    # Wrap with Coconut (adds hidden state feedback)
    model = Coconut(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = model.to(device)

    # Load data
    print("\nLoading data...")
    train_path = config.data.train_path
    val_path = config.data.val_path
    max_train = getattr(config.data, 'max_train_samples', 5000)
    max_val = getattr(config.data, 'max_val_samples', 500)

    # Tokenized datasets (HF Dataset)
    train_base = get_dataset(train_path, tokenizer, max_size=max_train)
    val_base = get_dataset(val_path, tokenizer, max_size=max_val)

    # Raw data for answer lookup during eval
    raw_val_data = json.load(open(val_path))[:max_val]

    print(f"  Train samples: {len(train_base)}")
    print(f"  Val samples: {len(val_base)}")

    # Training parameters
    c_thought = getattr(config.coconut, 'c_thought', 1)
    epochs_per_stage = getattr(config.coconut, 'epochs_per_stage', 3)
    max_stage = getattr(config.coconut, 'max_latent_stage', 3)
    reset_optimizer = getattr(config.coconut, 'reset_optimizer', True)

    batch_size = getattr(config.training, 'batch_size', 16)
    num_epochs = getattr(config.training, 'num_epochs', 20)
    lr = getattr(config.training, 'lr', 1e-4)
    weight_decay = getattr(config.training, 'weight_decay', 0.01)
    grad_accum = getattr(config.training, 'gradient_accumulation_steps', 1)

    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  c_thought: {c_thought}")
    print(f"  epochs_per_stage: {epochs_per_stage}")
    print(f"  max_stage: {max_stage}")
    print(f"  reset_optimizer: {reset_optimizer}")
    print("=" * 50)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)

    global_step = 0
    best_acc = 0
    prev_stage = -1

    for epoch in range(num_epochs):
        # Calculate current stage
        stage = min(epoch // epochs_per_stage, max_stage)
        num_latents = stage * c_thought

        # Reset optimizer when stage changes
        if stage != prev_stage and reset_optimizer and prev_stage >= 0:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"  [Stage change {prev_stage}->{stage}] Optimizer reset")
        prev_stage = stage

        print(f"\nEpoch {epoch+1}/{num_epochs} (Stage {stage}, {num_latents} latents)")

        # Build training dataset for current stage
        train_dataset = get_cot_latent_dataset(
            stage, train_base, config.coconut,
            start_id, latent_id, end_id,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

        # Training
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            loss = outputs.loss / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

        avg_loss = total_loss / num_batches
        print(f"  Avg loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % 2 == 0:
            model.eval()

            # Build eval dataset: question + latent tokens only
            val_dataset = get_question_latent_dataset(
                stage, val_base, config.coconut,
                start_id, latent_id, end_id,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, collate_fn=collator,
            )

            correct = 0
            total = 0

            print("  Evaluating...")
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc="Eval")):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,
                    )

                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_answer = output_text.split("#")[-1].strip()

                    true_answer = raw_val_data[i]["answer"]

                    if pred_answer == true_answer:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0
            print(f"  Val accuracy: {correct}/{total} = {acc:.2%}")

            # Save best model
            if acc > best_acc:
                best_acc = acc
                save_path = getattr(config.misc, 'save_path', './checkpoints')
                os.makedirs(save_path, exist_ok=True)

                # Save base model (Coconut is just a wrapper)
                model.base_causallm.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  Saved best model!")

    print(f"\nTraining done! Best accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
