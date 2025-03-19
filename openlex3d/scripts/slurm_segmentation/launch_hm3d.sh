#!/bin/bash

ALGORITHM=$1

for n in 1 5 10
do
    for sequence in 00824  00829  00843  00847  00873  00877  00890
    do 
        sbatch $HOME/openlex3d/openlex3d/scripts/slurm_segmentation/sbatch.sh -cp $HOME/openlex3d/openlex3d/config -cn eval_segmentation dataset=hm3d evaluation.algorithm=$ALGORITHM dataset.scene=$sequence evaluation.topn=$n model.device=cuda:0 
    done
done
