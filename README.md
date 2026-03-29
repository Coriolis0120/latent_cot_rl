# Latent CoT with RL

基于COCONUT的变长潜在思维链研究，使用强化学习学习何时停止思考。

## 快速开始

### 本地单卡训练

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; \
GPT2LMHeadModel.from_pretrained('openai-community/gpt2').save_pretrained('./models/gpt2'); \
GPT2Tokenizer.from_pretrained('openai-community/gpt2').save_pretrained('./models/gpt2')"

# 3. 训练
python scripts/train_dist.py --config configs/config.yaml
```

### 集群多卡训练 (Slurm)

```bash
# 提交作业
sbatch scripts/slurm_train.sh

# 或交互式运行
srun --gres=gpu:4 torchrun --nproc_per_node=4 scripts/train_dist.py --config configs/config_cluster.yaml
```

## 项目结构

```
latent_cot_rl/
├── CLAUDE.md              # 详细研究文档 ⭐
├── configs/
│   ├── config.yaml        # 基础配置
│   └── config_cluster.yaml # 集群配置
├── scripts/
│   ├── train_dist.py      # 分布式训练脚本
│   └── slurm_train.sh     # Slurm提交脚本
├── models/                # 模型存放目录
└── data/                  # 数据目录
```

## 核心思想

```
传统CoT:  问题 → "文字思考过程" → 答案
COCONUT:  问题 → [向量×固定长度] → 答案
我们:     问题 → [向量×动态长度] → 答案
                      ↑
                 RL决定何时停止
```

## Slurm配置说明

`scripts/slurm_train.sh` 中可以修改：

```bash
#SBATCH --gres=gpu:4      # GPU数量
#SBATCH --time=24:00:00   # 运行时间
#SBATCH --partition=gpu   # 分区名称
```

## 详细文档

请查看 [CLAUDE.md](CLAUDE.md) 获取完整的研究背景、方法设计和实验计划。
