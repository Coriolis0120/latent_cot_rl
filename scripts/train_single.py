#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single GPU training script
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


def load_data(path, tokenizer, max_size=10000):
    """Load and tokenize data"""
    data = json.load(open(path))[:max_size]

    processed = []
    for i, sample in enumerate(data):
        question = sample["question"] + "\n"
        steps = [s + "\n" for s in sample["steps"]]
        answer = "### " + sample["answer"] + tokenizer.eos_token

        q_tokens = tokenizer.encode(question, add_special_tokens=True)
        step_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in steps]
        a_tokens = tokenizer.encode(answer, add_special_tokens=False)

        processed.append({
            "question": q_tokens,
            "steps": step_tokens,
            "answer": a_tokens,
            "raw": sample,
            "idx": i,
        })

    return processed


class CoconutDataset(torch.utils.data.Dataset):
    """COCONUT dataset"""

    def __init__(self, data, num_latents, start_id, latent_id, end_id):
        self.data = data
        self.num_latents = num_latents
        self.start_id = start_id
        self.latent_id = latent_id
        self.end_id = end_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Build input sequence
        tokens = sample["question"].copy()
        tokens.append(self.start_id)
        tokens.extend([self.latent_id] * self.num_latents)
        tokens.append(self.end_id)

        # Add remaining steps
        skip_steps = min(self.num_latents, len(sample["steps"]))
        for step in sample["steps"][skip_steps:]:
            tokens.extend(step)

        # Add answer
        tokens.extend(sample["answer"])

        # Build labels (only compute loss on non-latent parts)
        labels = [-100] * len(sample["question"])
        labels.append(-100)
        labels.extend([-100] * self.num_latents)
        labels.append(-100)
        remaining_len = len(tokens) - len(labels)
        labels.extend(tokens[len(labels):])

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "idx": sample["idx"],
        }


def collate_fn(batch):
    """Collate function for dataloader"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    idxs = [item["idx"] for item in batch]

    # Padding
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_labels = []
    attention_mask = []

    for ids, lbls in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.tensor([-100] * pad_len)]))
        attention_mask.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_mask),
        "idxs": idxs,
    }


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
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    print(f"Special tokens: start={start_id}, end={end_id}, latent={latent_id}")

    # Initialize new token embeddings
    with torch.no_grad():
        ref_id = tokenizer.encode("<", add_special_tokens=False)[0]
        for tid in [start_id, end_id, latent_id]:
            model.transformer.wte.weight[tid] = model.transformer.wte.weight[ref_id]
            model.lm_head.weight[tid] = model.lm_head.weight[ref_id]

    model = model.to(device)

    # Load data
    print("\nLoading data...")
    train_path = config.data.train_path
    val_path = config.data.val_path
    max_train = getattr(config.data, 'max_train_samples', 5000)
    max_val = getattr(config.data, 'max_val_samples', 500)

    train_data = load_data(train_path, tokenizer, max_size=max_train)
    val_data = load_data(val_path, tokenizer, max_size=max_val)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")

    # Training parameters
    c_thought = getattr(config.coconut, 'c_thought', 1)
    epochs_per_stage = getattr(config.coconut, 'epochs_per_stage', 3)
    max_stage = getattr(config.coconut, 'max_latent_stage', 3)

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
    print("=" * 50)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    best_acc = 0

    for epoch in range(num_epochs):
        # Calculate current stage
        stage = min(epoch // epochs_per_stage, max_stage)
        num_latents = stage * c_thought

        print(f"\nEpoch {epoch+1}/{num_epochs} (Stage {stage}, {num_latents} latents)")

        # Create dataset
        train_dataset = CoconutDataset(train_data, num_latents, start_id, latent_id, end_id)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # Training
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Training")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
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
            val_dataset = CoconutDataset(val_data[:100], num_latents, start_id, latent_id, end_id)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, collate_fn=collate_fn
            )

            correct = 0
            total = 0

            print("  Evaluating...")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Eval"):
                    input_ids = batch["input_ids"].to(device)
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                    )

                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_answer = output_text.split("#")[-1].strip()

                    idx = batch["idxs"][0]
                    true_answer = val_data[idx]["raw"]["answer"]

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

                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
                print(f"  Saved best model!")

    print(f"\nTraining done! Best accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
