#!/bin/bash

for dataset in replica scannetpp hm3d
do
    for algorithm in concept-graphs openmask3d_nms_0.5_duplicated bare hovsg
    do
        for level in l0 l1 all
        do
            sbatch $HOME/openlex3d/openlex3d/scripts/slurm/query_sbatch.sh dataset=$dataset pred.method=$algorithm query.level=$level model.device=cuda:0
        done
    done
done
