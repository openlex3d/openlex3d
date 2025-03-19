#!/bin/bash

ALGORITHM=$1

for n in 1 5 10
do
    for sequence in room0 room1 room2 office0 office1 office2 office3 office4
    do 
        sbatch $HOME/openlex3d/openlex3d/scripts/slurm_segmentation/sbatch.sh -cp $HOME/openlex3d/openlex3d/config -cn eval_segmentation dataset=replica evaluation.algorithm=$ALGORITHM dataset.scene=$sequence evaluation.topn=$n model.device=cuda:0 
    done
done
