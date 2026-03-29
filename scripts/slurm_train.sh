#!/bin/bash
#SBATCH --job-name=latent_cot        # 作业名称
#SBATCH --nodes=1                    # 节点数
#SBATCH --ntasks-per-node=1          # 每个节点的任务数
#SBATCH --cpus-per-task=8            # 每个任务的CPU核数
#SBATCH --gres=gpu:4                 # GPU数量（根据实际情况调整）
#SBATCH --mem=64G                    # 内存
#SBATCH --time=24:00:00              # 最大运行时间
#SBATCH --output=logs/%j.out         # 标准输出日志
#SBATCH --error=logs/%j.err          # 错误日志
#SBATCH --partition=gpu              # 分区名称（根据集群调整）

# ========================================
# Latent CoT with RL - Slurm 训练脚本
# ========================================

# 创建日志目录
mkdir -p logs

# 打印作业信息
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "========================================"

# 加载环境（根据集群实际情况修改）
# module load cuda/12.0
# module load anaconda3
# source activate coconut

# 或者使用conda
# conda activate coconut

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# 配置
CONFIG_FILE=${1:-configs/config.yaml}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "========================================"

# 运行训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train_dist.py \
    --config $CONFIG_FILE

echo "========================================"
echo "End time: $(date)"
echo "========================================"
