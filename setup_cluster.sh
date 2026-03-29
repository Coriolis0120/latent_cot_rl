#!/bin/bash
# 集群环境安装脚本 - 使用现有module

set -e

echo "=========================================="
echo "  Latent CoT with RL - 集群环境配置"
echo "=========================================="

# 加载集群module
echo "[1/4] 加载集群module..."
module load miniconda3/23.5.2
module load cuda12.8/toolkit/12.8.1

# 初始化conda
eval "$(conda shell.bash hook)"

ENV_NAME="latent_cot"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 '${ENV_NAME}' 已存在"
else
    echo ""
    echo "[2/4] 创建conda环境..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# 激活环境
echo ""
echo "[3/4] 激活环境并安装依赖..."
conda activate ${ENV_NAME}

# 使用集群的CUDA安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm pyyaml accelerate wandb

# 下载模型
echo ""
echo "[4/4] 下载GPT-2模型..."
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
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF

echo "=========================================="
echo ""
echo "完成! 使用方法:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/train_single.py --config configs/config_5090.yaml"
echo ""
