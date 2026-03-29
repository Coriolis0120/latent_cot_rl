"""
主训练脚本

整合 COCONUT + RL 变长策略

训练流程:
1. 加载数据和模型
2. 对于每个问题:
   - 初始化状态
   - 循环:
     - RL策略决定 continue/stop
     - 如果continue: 生成latent, 更新状态
     - 如果stop: 预测答案, 计算奖励
3. 收集轨迹, 更新RL策略
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from typing import List, Tuple, Optional

from models.coconut import SimpleCoconut, CoconutWithRL
from models.rl_policy import StopPolicy, REINFORCETrainer


class ArithmeticDataset(torch.utils.data.Dataset):
    """算术数据集"""

    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """简单的collate函数"""
    return batch


class LatentCoTTrainer:
    """
    整合训练器
    """

    def __init__(
        self,
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.device = device

        # 初始化模型
        print("初始化模型...")
        self.coconut = CoconutWithRL(
            model_name=config['model']['name'],
            device=device,
        )

        # 初始化RL策略
        self.policy = StopPolicy(
            state_dim=128,  # 与coconut.state_encoder输出一致
        ).to(device)

        # 初始化RL训练器
        self.rl_trainer = REINFORCETrainer(
            policy=self.policy,
            lr=config['rl']['lr'],
            gamma=config['rl']['gamma'],
            entropy_coef=config['rl']['entropy_coef'],
            length_penalty=config['rl']['length_penalty'],
        )

        # 加载数据
        print("加载数据...")
        train_dataset = ArithmeticDataset(config['data']['train_path'])
        val_dataset = ArithmeticDataset(config['data']['val_path'])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # 逐个处理
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")

    def generate_episode(
        self,
        sample: dict,
        max_latents: int = 6,
        deterministic: bool = False,
    ) -> dict:
        """
        生成一个训练episode

        Args:
            sample: 数据样本 {question, steps, answer}
            max_latents: 最大latent数量
            deterministic: 是否使用确定性策略

        Returns:
            轨迹信息
        """
        question = sample['question']
        correct_answer = sample['answer']

        # 准备输入（问题 + start_latent）
        question_text = question + "\n"
        question_ids = self.coconut.tokenizer.encode(question_text, return_tensors='pt').to(self.device)

        # 获取问题的编码（作为初始状态）
        with torch.no_grad():
            outputs = self.coconut.model.transformer(question_ids)
            question_encoding = outputs.last_hidden_state[:, -1, :]  # [1, hidden]

        # RL交互循环
        states = []
        actions = []
        log_probs = []
        values = []

        current_latents = []
        num_latents = 0

        # 当前输入: 问题 + start_latent
        current_ids = torch.cat([
            question_ids,
            torch.tensor([[self.coconut.start_latent_id]], device=self.device)
        ], dim=1)

        while num_latents < max_latents:
            # 获取RL状态
            state = self.coconut.get_state_for_rl(question_encoding, current_latents)
            states.append(state.squeeze(0))  # [state_dim]

            # 策略选择动作
            policy_output = self.policy(state, deterministic=deterministic)
            actions.append(policy_output.action.item())
            log_probs.append(policy_output.log_prob.squeeze(0))
            if policy_output.value is not None:
                values.append(policy_output.value.squeeze(0))

            # 如果选择停止，退出循环
            if policy_output.action.item() == 1:  # stop
                break

            # 否则，生成一个latent
            with torch.no_grad():
                # 添加latent token
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[self.coconut.latent_id]], device=self.device)
                ], dim=1)

                # 获取latent的hidden state
                outputs = self.coconut.model.transformer(current_ids)
                latent_hidden = outputs.last_hidden_state[:, -1, :]

            current_latents.append(latent_hidden)
            num_latents += 1

        # 添加 end_latent 并生成答案
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[self.coconut.end_latent_id]], device=self.device)
        ], dim=1)

        # 生成答案
        with torch.no_grad():
            output_ids = self.coconut.model.generate(
                current_ids,
                max_new_tokens=20,
                pad_token_id=self.coconut.pad_id,
                eos_token_id=self.coconut.eos_id,
            )

        # 解码答案
        full_output = self.coconut.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_answer = full_output.split('=')[-1].strip().split()[0] if '=' in full_output else full_output.split()[-1]

        # 简单的答案匹配
        is_correct = self._check_answer(predicted_answer, correct_answer)

        # 计算奖励
        reward = self.rl_trainer.compute_reward(is_correct, num_latents)
        rewards = [0] * (len(actions) - 1) + [reward]  # 只有最后一步有奖励

        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'num_latents': num_latents,
            'is_correct': is_correct,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'question': question,
        }

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """检查答案是否正确"""
        # 提取数字
        import re
        pred_nums = re.findall(r'-?\d+', predicted)
        corr_nums = re.findall(r'-?\d+', correct)

        if pred_nums and corr_nums:
            return pred_nums[0] == corr_nums[0]
        return predicted.strip() == correct.strip()

    def train_epoch(self) -> dict:
        """训练一个epoch"""
        self.policy.train()

        trajectories = []
        total_correct = 0
        total_latents = 0

        for batch in tqdm(self.train_loader, desc="训练"):
            sample = batch[0]

            # 生成episode
            episode = self.generate_episode(
                sample,
                max_latents=self.config['latent']['max_thoughts'],
                deterministic=False,
            )

            trajectories.append(episode)
            total_correct += episode['is_correct']
            total_latents += episode['num_latents']

            # 每收集N个轨迹就更新一次
            if len(trajectories) >= self.config['training']['batch_size']:
                stats = self.rl_trainer.update(trajectories)
                trajectories = []

        # 处理剩余轨迹
        if trajectories:
            stats = self.rl_trainer.update(trajectories)

        return {
            **stats,
            'accuracy': total_correct / len(self.train_loader),
            'avg_latents': total_latents / len(self.train_loader),
        }

    def evaluate(self) -> dict:
        """评估"""
        self.policy.eval()

        total_correct = 0
        total_latents = 0
        examples = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="评估"):
                sample = batch[0]

                episode = self.generate_episode(
                    sample,
                    max_latents=self.config['latent']['max_thoughts'],
                    deterministic=True,  # 评估用确定性策略
                )

                total_correct += episode['is_correct']
                total_latents += episode['num_latents']

                # 保存前几个例子用于展示
                if len(examples) < 5:
                    examples.append({
                        'question': episode['question'],
                        'predicted': episode['predicted_answer'],
                        'correct': episode['correct_answer'],
                        'is_correct': episode['is_correct'],
                        'num_latents': episode['num_latents'],
                    })

        return {
            'accuracy': total_correct / len(self.val_loader),
            'avg_latents': total_latents / len(self.val_loader),
            'examples': examples,
        }

    def train(self, num_epochs: int, eval_every: int = 10):
        """完整训练流程"""
        print(f"\n开始训练，共 {num_epochs} epochs")
        print("=" * 50)

        best_accuracy = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练
            train_stats = self.train_epoch()
            print(f"训练 - 准确率: {train_stats['accuracy']:.2%}, "
                  f"平均latent数: {train_stats['avg_latents']:.1f}")

            # 评估
            if (epoch + 1) % eval_every == 0:
                eval_stats = self.evaluate()
                print(f"验证 - 准确率: {eval_stats['accuracy']:.2%}, "
                      f"平均latent数: {eval_stats['avg_latents']:.1f}")

                # 展示例子
                print("\n示例:")
                for i, ex in enumerate(eval_stats['examples'][:3]):
                    status = "✓" if ex['is_correct'] else "✗"
                    print(f"  [{status}] {ex['question']} "
                          f"→ 预测: {ex['predicted']}, "
                          f"正确: {ex['correct']}, "
                          f"latents: {ex['num_latents']}")

                # 保存最佳模型
                if eval_stats['accuracy'] > best_accuracy:
                    best_accuracy = eval_stats['accuracy']
                    self.save_checkpoint(f"best_model.pt")
                    print(f"保存最佳模型 (准确率: {best_accuracy:.2%})")

        print(f"\n训练完成! 最佳准确率: {best_accuracy:.2%}")

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.rl_trainer.optimizer.state_dict(),
        }, f"checkpoints/{filename}")

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.rl_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("配置:")
    print(yaml.dump(config, default_flow_style=False))

    # 初始化训练器
    trainer = LatentCoTTrainer(config)

    # 开始训练
    trainer.train(
        num_epochs=config['training']['num_episodes'] // len(trainer.train_loader),
        eval_every=config['training']['eval_every'],
    )


if __name__ == "__main__":
    main()
