#!/bin/bash
#SBATCH --job-name=ds_inference       
#SBATCH --output=ds_inference_%j.out   
#SBATCH --error=ds_inference_%j.err    
#SBATCH --nodelist=hpcgpu02 
#SBATCH --nodes=1               
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=6              
#SBATCH --gres=gpu:7                   
#SBATCH --mem=200G                    
#SBATCH --time=24:00:00              
#SBATCH --partition=gpu 

# module load cuda/12.2  
# module load python/3.10  
source ~/.my_envs_llm_cu121/env.sh
cd /home/mail/2021r1/r130026060/ds
echo "Starting DeepSpeed inference on node $SLURMD_NODENAME with $SLURM_NTASKS tasks..."

srun python inference_ds_v5.py

echo "Inference finished."
