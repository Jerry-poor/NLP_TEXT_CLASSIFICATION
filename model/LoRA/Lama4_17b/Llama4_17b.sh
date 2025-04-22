#!/bin/bash
#SBATCH --job-name=Llama4_17b_Qlora
#SBATCH --nodes=5                             # 单节点
#SBATCH --ntasks-per-node=8                  # 每节点进程数，通常等于GPU数
#SBATCH --gres=gpu:8                          # 每节点使用4块GPU
#SBATCH --cpus-per-task=6                    # 每个任务使用6个CPU
#SBATCH --mem=400G                             # 每节点内存
#SBATCH --time=240:00:00                       # 作业最大时间
#SBATCH --partition=gpu                       # 根据你的集群配置选择实际可用的分区（如 gpu/compute 等）


source ~/.my_envs_llm_cu121/env.sh
# 切换到包含 ds.py 的工作目录
cd /home/mail/2021r1/r130026060
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# ============ 4. 动态检测实际可用 GPU 数量 ============
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
echo "Detected usable GPU count: $GPU_COUNT"
export MASTER_PORT=$((10000 + RANDOM % 50000))
echo "Using MASTER_PORT=$MASTER_PORT"
# ============ 3. 启动 DeepSpeed 训练 ============
echo "Starting DeepSpeed training..." 
deepspeed Llama4_17b.py --deepspeed Llama4_17b.json  --config axolotl_config.yaml


echo "DeepSpeed training finished."

null
