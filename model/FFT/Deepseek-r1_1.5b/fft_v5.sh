#!/bin/bash
#SBATCH --job-name=deepspeed_llm
#SBATCH --output=logs/fft_run_%j.out          # 日志输出路径，请提前创建 logs 目录
#SBATCH --error=logs/fft_run_%j.err
#SBATCH --nodes=1                             # 单节点
#SBATCH --ntasks-per-node=4                  # 启动 7 个进程
#SBATCH --gres=gpu:4                          # 分配 7 张 GPU
#SBATCH --cpus-per-task=12                    # 每个任务使用 6 个 CPU
#SBATCH --mem=256G                            # 每节点内存
#SBATCH --time=72:00:00                       # 作业最大时间
#SBATCH --partition=gpu                       # 分区名称
#SBATCH --nodelist=hpcgpu01
# ============ 1. 加载用户环境 ============
source $HOME/.my_envs_llm_cu121/env.sh

# ============ 2. 切换到工作目录 ============
cd /home/mail/2021r1/r130026060/ds

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# （可选）动态检测实际可用 GPU 数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
echo "Detected usable GPU count: $GPU_COUNT"

# ============ 4. 分布式通信 & 运行时优化 ============
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + RANDOM % 50000))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# ============ 5. 启动 DeepSpeed 训练 ============
echo "Starting DeepSpeed training..."
deepspeed ds_v5.py --deepspeed ds_sp.json

echo "DeepSpeed training finished."
