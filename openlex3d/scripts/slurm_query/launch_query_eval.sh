#!/bin/bash

DATASET=$1
ALGORITHM=$2

sbatch $HOME/openlex3d/openlex3d/scripts/slurm/query_sbatch.sh dataset=$DATASET pred.method=$ALGORITHM model.device=cuda:0
