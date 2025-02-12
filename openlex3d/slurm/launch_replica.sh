#!/bin/bash

ALGORITHM=$1

for n in 1
do
    # for sequence in room0 room1 room2 office0 office1 office2 office3 office4
    for sequence in room0
    do 
        sbatch $HOME/openlex3d/openlex3d/slurm/sbatch.sh -cp $HOME/openlex3d/openlex3d/config -cn replica model.device=cuda:0 evaluation.algorithm=$ALGORITHM dataset.scene=$sequence evaluation.top_n=$n
    done
done