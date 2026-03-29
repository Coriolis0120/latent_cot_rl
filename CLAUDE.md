# Latent CoT with RL - 研究项目

> 基于COCONUT的变长潜在思维链研究，使用强化学习学习何时停止思考

---

## 项目概述

### 研究问题

传统思维链（Chain of Thought, CoT）使用自然语言进行推理，但存在以下问题：
1. 效率低：需要生成大量文字
2. 冗余：很多token只是为了语句通顺
3. 受限于语言：有些推理用语言表达不清楚

**我们的目标**：用连续向量代替自然语言进行推理，并用RL学习最优的思考长度。

### 核心思想

```
传统CoT:  问题 q → "思考过程(文字)" → 答案 a
COCONUT:  问题 q → [向量×k固定] → 答案 a
我们:     问题 q → [向量×动态长度] → 答案 a
                      ↑
                 RL决定何时停止
```

---

## 数学形式化

### 基本定义

```
P(a|q)     : 没有CoT时，模型预测答案a的概率
P(a|q,x)   : 有CoT x时，模型预测答案a的概率
x          : 思维链（可以是自然语言或连续向量）
```

### 优化目标

```
x* = argmax_x P(a* | q, x)

其中:
- q: 问题
- a*: 正确答案
- x: 连续向量的思维链
```

### 变长问题

```
x ∈ R^{ℓ×d}  其中 ℓ 是变量，取决于问题难度

简单问题: ℓ 小 (少想几步)
复杂问题: ℓ 大 (多想几步)
```

---

## 相关工作：COCONUT

### 论文信息

- **标题**: Training Large Language Models to Reason in a Continuous Latent Space
- **机构**: Meta (FAIR)
- **会议**: ICLR 2025
- **代码**: https://github.com/facebookresearch/coconut

### COCONUT核心方法

#### 1. Latent Token机制

```python
# 不解码成文字，直接用hidden state作为下一个输入
hidden_state = model.forward(input_ids)  # 最后隐藏层
# 不做: next_token = decode(hidden_state)
# 而是: next_input = hidden_state  # 直接作为下一个输入embedding
```

#### 2. 特殊Token

```
<|start-latent|>  : 开始潜在思考
<|latent|>        : 一个潜在向量
<|end-latent|>    : 结束潜在思考
```

#### 3. 渐进式训练

```
Stage 0: Q → "步骤1" → "步骤2" → "步骤3" → A  (全部文字)
Stage 1: Q → [L] → "步骤2" → "步骤3" → A      (替换1步)
Stage 2: Q → [L] → [L] → "步骤3" → A          (替换2步)
Stage 3: Q → [L] → [L] → [L] → A              (全部latent)
```

#### 4. 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| c_thought | 每个推理步骤用几个latent | 1-2 |
| epochs_per_stage | 每个stage训练几个epoch | 3-5 |
| max_latent_stage | 最多替换几个步骤 | 3-6 |

### COCONUT的局限性

1. **训练笨拙**：需要N个stage，每个stage单独训练
2. **长度固定**：训练后latent数量固定，不能自适应
3. **不是直接优化**：是渐进替换，不是argmax
4. **不可解释**：不知道latent编码了什么

---

## 我们的创新点

### 1. 变长Latent (核心)

```
COCONUT: 所有问题用固定k个latent
我们:    RL学习何时停止，不同问题不同长度
```

### 2. RL学习停止策略

把"何时停止"建模为强化学习问题：

| RL概念 | 我们的问题 |
|--------|-----------|
| 状态 s_t | (问题编码, 已生成的latent序列) |
| 动作 a_t | Continue / Stop |
| 奖励 r | 答案正确性 + 长度惩罚 |

### 3. 可解释性探索

尝试将latent解码回自然语言，理解它编码了什么：

```
latent x → decoder → "3-2=1, 然后+5=6"
```

---

## 方法设计

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   输入: 问题 q                                       │
│          ↓                                          │
│   ┌──────────────────┐                              │
│   │  Base LLM        │                              │
│   │  (GPT-2)         │                              │
│   └────────┬─────────┘                              │
│            ↓                                        │
│   ┌──────────────────┐    ┌─────────────────┐      │
│   │  RL Policy Net   │───→│ Action: Stop?   │      │
│   │  (2层MLP)        │    │ Continue?       │      │
│   └────────┬─────────┘    └─────────────────┘      │
│            │                     │                  │
│            ↓                     ↓                  │
│   ┌──────────────────┐    ┌─────────────────┐      │
│   │ 生成下一个        │    │ 输出答案 a       │      │
│   │ latent x_t       │    │                 │      │
│   └──────────────────┘    └─────────────────┘      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### RL策略网络

```python
class RLPolicy(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # [continue, stop] 概率
        )

    def forward(self, state):
        # state: 当前hidden state
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def select_action(self, state):
        probs = self.forward(state)
        action = torch.multinomial(probs, 1)  # 采样
        return action  # 0=continue, 1=stop
```

### 奖励函数

```python
def compute_reward(predicted_answer, correct_answer, num_latents):
    # 1. 正确性奖励
    if predicted_answer == correct_answer:
        r_correct = +1.0
    else:
        r_correct = 0.0

    # 2. 效率惩罚（鼓励简洁）
    r_efficiency = -0.05 * num_latents

    # 3. 总奖励
    reward = r_correct + r_efficiency

    return reward
```

### 训练算法

使用REINFORCE或PPO：

```python
def train_reinforce(model, policy, question, correct_answer):
    # 1. 生成轨迹
    latents = []
    log_probs = []
    state = model.encode(question)

    for t in range(max_steps):
        # 策略决定动作
        action, log_prob = policy.select_action(state)
        log_probs.append(log_prob)

        if action == STOP:
            break

        # 生成下一个latent
        latent = model.generate_latent(state)
        latents.append(latent)
        state = model.update_state(state, latent)

    # 2. 预测答案
    answer = model.predict(question, latents)

    # 3. 计算奖励
    reward = compute_reward(answer, correct_answer, len(latents))

    # 4. 更新策略
    loss = -sum(log_probs) * reward
    loss.backward()
    optimizer.step()

    return reward, len(latents)
```

---

## 实验设计

### 数据集

| 数据集 | 类型 | 样本数 | 难度 |
|--------|------|--------|------|
| 简单算术 | 数学运算 | 1000 | 简单 |
| ProsQA | 逻辑推理 | 30K | 中等 |
| ProntoQA | 逻辑推理 | 10K | 中等 |
| GSM8K | 数学应用题 | 7K | 困难 |

### 数据格式

```json
{
  "question": "问题内容",
  "answer": "答案",
  "steps": ["步骤1", "步骤2", "步骤3"]
}
```

### 对比方法

| 方法 | 描述 |
|------|------|
| No CoT | 直接预测，不思考 |
| Standard CoT | 自然语言思维链 |
| COCONUT (k=2) | 固定2个latent |
| COCONUT (k=4) | 固定4个latent |
| **Ours (RL)** | RL决定latent数量 |

### 评估指标

#### 主要指标

```python
# 1. 准确率
accuracy = correct_predictions / total_predictions

# 2. 平均latent数量
avg_latents = sum(num_latents) / total_predictions

# 3. 效率分数
efficiency = accuracy / avg_latents
```

#### 长度匹配评估

```python
# 对每个问题，暴力搜索最优长度
def find_optimal_length(question, correct_answer):
    results = []
    for l in range(1, max_len + 1):
        answer = model.predict(question, fixed_length=l)
        results.append({
            'length': l,
            'correct': (answer == correct_answer)
        })

    # 最短的正确长度
    optimal_l = min([r['length'] for r in results if r['correct']])
    return optimal_l

# 评估RL是否学到了最优长度
def evaluate_rl_length():
    for q, a in test_set:
        l_rl = rl_policy.decide_length(q)
        l_optimal = find_optimal_length(q, a)

        gap = l_rl - l_optimal  # 越小越好
        # gap > 0: 过度思考
        # gap < 0: 思考不足
```

### 实验流程

```
阶段1: 复现COCONUT
├── 运行官方代码
├── 理解训练过程
└── 获得baseline结果

阶段2: 实现RL策略
├── 设计policy网络
├── 实现REINFORCE算法
└── 验证能学到stop策略

阶段3: 对比实验
├── 固定长度 vs 变长
├── 不同难度问题的表现
└── 效率分析

阶段4: 可解释性分析
├── 尝试解码latent
├── 分析latent长度与问题复杂度的关系
└── 案例研究
```

---

## 代码结构

```
latent_cot_rl/
├── CLAUDE.md              # 本文档
├── README.md              # 项目说明
├── requirements.txt       # 依赖
│
├── configs/
│   └── config.yaml        # 配置文件
│
├── data/
│   └── arithmetic/        # 算术数据集
│
├── models/
│   ├── coconut.py         # COCONUT核心实现
│   ├── rl_policy.py       # RL策略网络
│   └── utils.py           # 工具函数
│
└── scripts/
    ├── generate_data.py   # 生成数据
    ├── train.py           # 训练主程序
    ├── train_rl.py        # RL训练
    └── eval.py            # 评估脚本
```

---

## 硬件要求

### 最低配置

| 组件 | 要求 |
|------|------|
| GPU | 12GB显存 (如Tesla P40) |
| 内存 | 16GB |
| 存储 | 10GB |

### 推荐配置

| 组件 | 要求 |
|------|------|
| GPU | 24GB显存 (如RTX 3090/A5000) |
| 内存 | 32GB |
| 存储 | 20GB |

### 当前资源

- Tesla P40 (24GB显存) × 1
- Pascal架构，无Tensor Core
- 训练速度较慢，但足够验证想法

---

## 运行指南

### 环境准备

```bash
# 创建环境
conda create -n coconut python=3.10
conda activate coconut

# 安装依赖
pip install torch transformers datasets tqdm pyyaml wandb
```

### 下载模型

```bash
# 方法1: huggingface-cli
huggingface-cli download openai-community/gpt2 --local-dir ./models/gpt2

# 方法2: Python脚本
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
GPT2LMHeadModel.from_pretrained('openai-community/gpt2').save_pretrained('./models/gpt2')
GPT2Tokenizer.from_pretrained('openai-community/gpt2').save_pretrained('./models/gpt2')
"
```

### 运行COCONUT

```bash
# 简化版脚本
python run_simple.py --config args/test_simple.yaml

# 原版脚本（需要分布式）
torchrun --nnodes=1 --nproc_per_node=1 run.py args/config.yaml
```

### 运行RL训练（待实现）

```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

---

## 时间线

### Week 1-2: 环境搭建与复现
- [x] 阅读COCONUT论文和代码
- [ ] 在P40上部署COCONUT
- [ ] 跑通baseline实验
- [ ] 记录实验结果

### Week 3-4: RL策略实现
- [ ] 设计policy网络
- [ ] 实现REINFORCE算法
- [ ] 在小数据集上验证
- [ ] 调试和优化

### Week 5-6: 实验与分析
- [ ] 完整对比实验
- [ ] 可解释性分析
- [ ] 撰写报告

---

## 参考文献

1. **COCONUT**: Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space" (ICLR 2025)
2. **Chain-of-Thought**: Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022)
3. **REINFORCE**: Williams. "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (1992)
4. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

---

## 联系方式

- 项目负责人: [你的名字]
- 导师: [导师名字]
- 开始日期: 2026年3月

---

*最后更新: 2026年3月29日*
