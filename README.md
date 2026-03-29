# Latent CoT with RL

基于COCONUT的变长潜在思维链研究，使用强化学习学习何时停止思考。

## 禂述

```
传统CoT:  问题 → "文字思考过程" → 答案
COCONUT:  问题 → [向量×固定长度] → 答案
我们:     问题 → [向量×动态长度] → 答案
                      ↑
                 RL决定何时停止
```

## 环境配置 (Conda)

```bash
# 克隆仓库
git clone git@github.com:Coriolis0120/latent_cot_rl.git
cd latent_cot_rl

# 运行安装脚本
bash setup.sh

# 激活环境
conda activate latent_cot
```

## 训练

### 单卡训练 (推荐)

```bash
conda activate latent_cot
python scripts/train_single.py --config configs/config_5090.yaml
```

### 多卡训练 (如需要)

```bash
conda activate latent_cot
torchrun --nproc_per_node=4 scripts/train_dist.py --config configs/config_cluster.yaml
```

## 项目结构

```
latent_cot_rl/
├── CLAUDE.md              # 详细研究文档 ⭐
├── environment.yml        # Conda环境配置
├── setup.sh               # 一键安装脚本
├── configs/
│   ├── config_5090.yaml   # 单卡5090配置
│   └── config_cluster.yaml # 多卡集群配置
├── scripts/
│   ├── train_single.py    # 单卡训练
│   └── train_dist.py      # 分布式训练
└── models/                # 模型存放目录
```

## 硬件要求

| 配置 | GPU | 显存 | Batch Size |
|------|-----|------|------------|
| 最小 | 任意 | 12GB | 4 |
| 推荐 | 5090 | 32GB | 32 |
| 集群 | 4×5090 | 128GB | 64 |

## 详细文档

请查看 [CLAUDE.md](CLAUDE.md) 获取完整的研究背景、方法设计和实验计划。
