#!/bin/bash -l

#SBATCH -J CS687
#SBATCH --time=150:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-sasan
#SBATCH --gres=gpu:2
#SBATCH -o CS687-%j.output
#SBATCH -e CS687-%j.err

module load conda3
module load cuda/11.8.0
module load module-git

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate py310

huggingface-cli whoami

python GRPOTrainer.py

conda deactivate