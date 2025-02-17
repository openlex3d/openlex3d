#!/bin/bash

ALGORITHM=$1

for n in 1 5 10
do
    for sequence in 0a76e06478  0a7cc12c0e  1f7cbbdde1  49a82360aa  4c5c60fa76  8a35ef3cfe  c0f5742640  fd361ab85f
    do 
        sbatch $HOME/openlex3d/openlex3d/slurm/sbatch.sh -cp $HOME/openlex3d/openlex3d/config -cn scannetpp model.device=cuda:0 evaluation.algorithm=$ALGORITHM dataset.scene=$sequence evaluation.top_n=$n
    done
done