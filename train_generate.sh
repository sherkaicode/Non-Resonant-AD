#!/bin/bash
#SBATCH --job-name=generate_train
#SBATCH --partition=tartarus
#SBATCH --output=generate_train.out   # stdout (%j = job ID)
#SBATCH --error=generate_train.err    # stderr
#SBATCH --ntasks=29
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=uranus

cd /home/aegis/Titan1/NRAD/data

python3 train_generate.py