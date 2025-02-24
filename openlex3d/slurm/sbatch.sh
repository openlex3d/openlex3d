#!/bin/bash

# Parameters
#SBATCH --job-name=openlex
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-03:00:00
#SBATCH --partition=long
#SBATCH -o /network/scratch/k/kumaraditya.gupta/logs/openlex-query-eval/slurm-%j.out  # Write the log on scratch


module load anaconda/3
conda activate o3d
cd $HOME/openlex3d

python -m openlex3d.scripts.evaluate_queries "$@"
