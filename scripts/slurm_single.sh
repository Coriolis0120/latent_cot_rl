#!/bin/bash
#SBATCH --job-name=latent_cot        # 作业名称
#SBATCH --output=logs/%j.out          # 输出日志
#SBATCH --error=logs/%j.err           # 错误日志
#SBATCH --gres=gpu:1                 # 1块GPU (单卡5090)
#SBATCH --cpus-per-task=8            # CPU核心数
#SBATCH --mem=32G                    # 内存
#SBATCH --time=24:00:00              # 最长运行时间
#SBATCH --partition=gpu              # GPU分区名称 (根据集群修改)

# 创建日志目录
mkdir -p logs

# 加载module
module load miniconda3/23.5.2
module load cuda12.8/toolkit/12.8.1

# 初始化conda
eval "$(conda shell.bash hook)"
conda activate latent_cot

# 打印环境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

nvidia-smi

# 运行训练
python scripts/train_single.py --config configs/config_5090.yaml

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
