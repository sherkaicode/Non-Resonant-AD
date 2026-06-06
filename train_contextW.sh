#!/bin/bash
#SBATCH --job-name=context_weight_train
#SBATCH --partition=tartarus
#SBATCH --output=context_weight_train.out   # stdout (%j = job ID)
#SBATCH --error=context_weight_train.err    # stderr
#SBATCH --ntasks=14
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=uranus

cd /home/aegis/Titan1/NRAD/data

python3 train_contextW.py