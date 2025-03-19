#!/bin/bash

# Parameters
#SBATCH --job-name=openlex
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-03:00:00
#SBATCH --partition=long
#SBATCH -o /path_to_logs/slurm-%j.out  # Write the log on scratch


module load python/3.10
export SSL_CERT_DIR=/etc/ssl/certs
source $HOME/venvs/openlex3d/bin/activate

ol3_evaluate "$@"
