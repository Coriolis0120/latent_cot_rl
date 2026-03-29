# Latent CoT with RL

基于COCONUT的变长潜在思维链研究，使用强化学习学习何时停止思考。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成数据
python scripts/generate_data.py

# 3. 训练
python scripts/train.py configs/config.yaml
```

## 项目结构

```
latent_cot_rl/
├── CLAUDE.md          # 详细研究文档 ⭐
├── configs/           # 配置文件
├── data/              # 数据文件
├── models/            # 模型代码
└── scripts/           # 训练脚本
```

## 核心思想

```
传统CoT:  问题 → "文字思考过程" → 答案
COCONUT:  问题 → [向量×固定长度] → 答案
我们:     问题 → [向量×动态长度] → 答案
                      ↑
                 RL决定何时停止
```

## 详细文档

请查看 [CLAUDE.md](CLAUDE.md) 获取完整的研究背景、方法设计和实验计划。
