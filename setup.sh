#!/bin/bash
# 环境安装脚本 - 简化版

set -e

echo "=========================================="
echo "  Latent CoT with RL - 环境配置"
echo "=========================================="

ENV_NAME="latent_cot"

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "错误: 请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 初始化conda
eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo ""
    echo "[1/3] 创建conda环境..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# 激活并安装依赖
echo ""
echo "[2/3] 安装依赖..."
conda activate ${ENV_NAME}

# 安装核心包
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm pyyaml accelerate wandb

# 下载模型
echo ""
echo "[3/3] 下载GPT-2模型..."
mkdir -p models/gpt2

python << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

path = "models/gpt2"
if not os.path.exists(f"{path}/config.json"):
    print("下载模型中...")
    model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"模型已保存到 {path}/")
else:
    print("模型已存在，跳过下载")
EOF

# 检查
echo ""
echo "=========================================="
python << 'EOF'
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

echo "=========================================="
echo ""
echo "完成! 使用方法:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/train_single.py --config configs/config_5090.yaml"
echo ""
