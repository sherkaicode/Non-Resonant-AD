#!/bin/bash
#SBATCH --job-name=split_data
#SBATCH --partition=tartarus
#SBATCH --output=split_data.out   # stdout (%j = job ID)
#SBATCH --error=split_data.err    # stderr
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

# Move into project directory
cd /home/aegis/Titan1/NRAD/data

# Run Python script
python3 regions_data.py
