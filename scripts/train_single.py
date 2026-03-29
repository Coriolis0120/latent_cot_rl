#!/usr/bin/env python3
"""
单卡训练脚本 - 适用于单块5090
用法:
    python scripts/train_single.py --config configs/config_5090.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import random
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(self, d):
        for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)


def load_data(path, tokenizer, max_size=10000):
    """加载并tokenize数据"""
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


class CoconuDataset(torch.utils.data.Dataset):
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

        tokens = sample["question"].copy()
        tokens.append(self.start_id)
        tokens.extend([self.latent_id] * self.num_latents)
        tokens.append(self.end_id)

        skip_steps = self.num_latents
        for step in sample["steps"][skip_steps:]:
            tokens.extend(step)

        tokens.extend(sample["answer"])

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
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

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
        "idxs": [item["idx"] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_5090.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = Config(yaml.safe_load(f))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    set_seed(config.misc.seed)

    # 加载模型
    model_id = config.model.model_id
    print(f"\n加载模型: {model_id}")

    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 添加特殊token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    print(f"特殊token: start={start_id}, end={end_id}, latent={latent_id}")

    # 初始化新token embedding
    with torch.no_grad():
        ref_id = tokenizer.encode("<", add_special_tokens=False)[0]
        for tid in [start_id, end_id, latent_id]:
            model.transformer.wte.weight[tid] = model.transformer.wte.weight[ref_id]
            model.lm_head.weight[tid] = model.lm_head.weight[ref_id]

    model = model.to(device)

    # 加载数据
    print("\n加载数据...")
    train_data = load_data(config.data.train_path, tokenizer, config.data.max_train_samples)
    val_data = load_data(config.data.val_path, tokenizer, config.data.max_val_samples)
    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # 训练参数
    c_thought = config.coconut.c_thought
    epochs_per_stage = config.coconut.epochs_per_stage
    max_stage = config.coconut.max_latent_stage
    batch_size = config.training.batch_size
    grad_accum = config.training.gradient_accumulation_steps
    num_epochs = config.training.num_epochs

    print(f"\n训练配置:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  c_thought: {c_thought}")
    print(f"  epochs_per_stage: {epochs_per_stage}")
    print(f"  max_stage: {max_stage}")
    print("=" * 50)

    global_step = 0
    best_acc = 0

    for epoch in range(num_epochs):
        stage = min(epoch // epochs_per_stage, max_stage)
        num_latents = stage * c_thought

        print(f"\nEpoch {epoch+1}/{num_epochs} (Stage {stage}, {num_latents} latents)")

        # 创建数据集
        train_dataset = CoconuDataset(train_data, num_latents, start_id, latent_id, end_id)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # 训练
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
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
        print(f"平均损失: {avg_loss:.4f}")

        # 验证
        if (epoch + 1) % config.evaluation.eval_every == 0:
            model.eval()
            val_dataset = CoconuDataset(val_data[:100], num_latents, start_id, latent_id, end_id)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, collate_fn=collate_fn
            )

            correct = 0
            total = 0

            print("验证中...")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
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
            print(f"验证准确率: {correct}/{total} = {acc:.2%}")

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                save_path = config.misc.save_path
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
                print(f"保存最佳模型: {save_path}/best_model.pt")

    print(f"\n训练完成! 最佳准确率: {best_acc:.2%}")


if __name__ == "__main__":
    main()
