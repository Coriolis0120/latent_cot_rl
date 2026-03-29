#!/bin/bash
# 环境安装脚本

set -e

echo "=========================================="
echo "  Latent CoT with RL - 环境配置"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 请先安装 Anaconda 或 Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 创建环境
echo ""
echo "[1/4] 创建conda环境..."
conda env create -f environment.yml || conda env update -f environment.yml

# 激活环境
echo ""
echo "[2/4] 激活环境..."
eval "$(conda shell.bash hook)"
conda activate latent_cot

# 下载模型
echo ""
echo "[3/4] 下载GPT-2模型..."
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.makedirs('models/gpt2', exist_ok=True)
print('下载模型中...')
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
model.save_pretrained('models/gpt2')
tokenizer.save_pretrained('models/gpt2')
print('模型已保存到 models/gpt2/')
"

# 检查安装
echo ""
echo "[4/4] 检查安装..."
python -c "
import torch
import transformers
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=========================================="
echo "  安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate latent_cot"
echo "  python scripts/train_single.py --config configs/config_5090.yaml"
echo ""
