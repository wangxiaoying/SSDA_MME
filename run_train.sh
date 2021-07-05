#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=00-03:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-jnwang
hostname
nvidia-smi
module load python/3.8.10 
source ./venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python main.py --method $1 --dataset office_home --source $3 --target $4 --net $2 --save_check
