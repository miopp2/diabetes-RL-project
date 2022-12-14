#!/bin/bash

# SLURM Settings
#SBATCH --job-name="PPO_tensorflow"
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2
#SBATCH --mail-user=mirjam.oppliger@students.unibe.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Load Anaconda3
module load Anaconda3
eval "$(conda shell.bash hook)"

# Load your environment
conda activate simgluc

# Run your code
srun python3 ./train/simglucose_with_ins_obs_train_ppo_tensorflow.py
