#!/bin/bash
#SBATCH --job-name=deepspeed_llm
#SBATCH --output=logs/fft_run_%j.out        
#SBATCH --error=logs/fft_run_%j.err
#SBATCH --nodes=1                            
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:4                         
#SBATCH --cpus-per-task=48                  
#SBATCH --mem=128G                            
#SBATCH --time=24:00:00                       
#SBATCH --partition=gpu                       

#加载环境
source $HOME/.my_envs

#切换目录
cd xxx
export WANDB_API_KEY="xxx"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting DeepSpeed training..."
deepspeed ds.py --deepspeed ds_sp.json
echo "DeepSpeed training finished."



