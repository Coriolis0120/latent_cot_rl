#!/usr/bin/env python3
"""
分布式训练脚本 - 支持多GPU
用法:
    单机多卡: torchrun --nproc_per_node=4 scripts/train_dist.py --config configs/config.yaml
    Slurm: sbatch scripts/slurm_train.sh
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ:
        # Slurm 或 torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单GPU
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


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
    import json
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
    """COCONUT数据集"""

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

        # 构建输入序列
        tokens = sample["question"].copy()
        tokens.append(self.start_id)
        tokens.extend([self.latent_id] * self.num_latents)
        tokens.append(self.end_id)

        # 添加剩余步骤
        skip_steps = self.num_latents
        for step in sample["steps"][skip_steps:]:
            tokens.extend(step)

        # 添加答案
        tokens.extend(sample["answer"])

        # 构建labels
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
    """批处理函数"""
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


class CoconutModel(torch.nn.Module):
    """简化的COCONUT模型包装"""

    def __init__(self, base_model, latent_id):
        super().__init__()
        self.base_model = base_model
        self.latent_id = latent_id

    def forward(self, input_ids, labels, attention_mask=None, **kwargs):
        # 这里简化处理，直接用base model
        # 完整版需要处理latent token的替换
        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        return outputs

    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        return self.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=kwargs.get('pad_token_id', 50256),
            do_sample=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = Config(yaml.safe_load(f))

    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    is_main_process = (rank == 0)

    if is_main_process:
        print(f"配置:")
        print(f"  世界大小: {world_size}")
        print(f"  当前rank: {rank}")
        print(f"  设备: {device}")

    set_seed(config.misc.seed if hasattr(config, 'misc') else 42)

    # 加载模型
    model_id = config.model.model_id if hasattr(config.model, 'model_id') else "openai-community/gpt2"

    if is_main_process:
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

    if is_main_process:
        print(f"特殊token: start={start_id}, end={end_id}, latent={latent_id}")

    # 初始化新token embedding
    with torch.no_grad():
        ref_id = tokenizer.encode("<", add_special_tokens=False)[0]
        for tid in [start_id, end_id, latent_id]:
            model.transformer.wte.weight[tid] = model.transformer.wte.weight[ref_id]
            model.lm_head.weight[tid] = model.lm_head.weight[ref_id]

    # 包装模型
    model = CoconutModel(model, latent_id)
    model = model.to(device)

    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 加载数据
    if is_main_process:
        print("加载数据...")

    train_path = config.data.train_path if hasattr(config.data, 'train_path') else "data/prosqa_train.json"
    val_path = config.data.val_path if hasattr(config.data, 'val_path') else "data/prosqa_valid.json"
    max_train = config.data.max_train_samples if hasattr(config.data, 'max_train_samples') else 5000
    max_val = config.data.max_val_samples if hasattr(config.data, 'max_val_samples') else 500

    train_data_raw = load_data(train_path, tokenizer, max_size=max_train)
    val_data_raw = load_data(val_path, tokenizer, max_size=max_val)

    if is_main_process:
        print(f"  训练样本: {len(train_data_raw)}")
        print(f"  验证样本: {len(val_data_raw)}")

    # 优化器
    lr = config.training.lr if hasattr(config.training, 'lr') else 1e-4
    weight_decay = config.training.weight_decay if hasattr(config.training, 'weight_decay') else 0.01

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 训练参数
    num_epochs = config.training.num_epochs if hasattr(config.training, 'num_epochs') else 10
    batch_size = config.training.batch_size if hasattr(config.training, 'batch_size') else 8
    grad_accum = config.training.gradient_accumulation_steps if hasattr(config.training, 'gradient_accumulation_steps') else 1

    c_thought = config.coconut.c_thought if hasattr(config, 'coconut') else 1
    epochs_per_stage = config.coconut.epochs_per_stage if hasattr(config, 'coconut') else 2
    max_stage = config.coconut.max_latent_stage if hasattr(config, 'coconut') else 3

    if is_main_process:
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
        # 计算当前stage
        stage = min(epoch // epochs_per_stage, max_stage)
        num_latents = stage * c_thought

        # 创建数据集
        train_dataset = CoconuDataset(train_data_raw, num_latents, start_id, latent_id, end_id)

        # 分布式sampler
        sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # 训练
        model.train()
        if sampler:
            sampler.set_epoch(epoch)

        total_loss = 0
        num_batches = 0

        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Stage {stage})")
        else:
            pbar = train_loader

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

            if is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

        avg_loss = total_loss / num_batches

        if is_main_process:
            print(f"\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

        # 验证
        if (epoch + 1) % 2 == 0 and is_main_process:
            model.eval()
            val_dataset = CoconuDataset(val_data_raw[:100], num_latents, start_id, latent_id, end_id)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, collate_fn=collate_fn
            )

            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    input_ids = batch["input_ids"].to(device)
                    outputs = model.generate(input_ids=input_ids, max_new_tokens=20)

                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred_answer = output_text.split("#")[-1].strip()

                    idx = batch["idxs"][0]
                    true_answer = val_data_raw[idx]["raw"]["answer"]

                    if pred_answer == true_answer:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0
            print(f"验证准确率: {correct}/{total} = {acc:.2%}")

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                save_path = config.misc.save_path if hasattr(config, 'misc') else "./checkpoints"
                os.makedirs(save_path, exist_ok=True)

                # 保存模型（只保存主进程的）
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(save_path, f"best_model.pt")
                )
                print(f"保存最佳模型: {save_path}/best_model.pt")

    if is_main_process:
        print(f"\n训练完成! 最佳准确率: {best_acc:.2%}")

    # 清理
    cleanup_distributed()


if __name__ == "__main__":
    main()
