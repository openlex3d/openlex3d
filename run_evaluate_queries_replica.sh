#!/bin/bash

python -m openlex3d.scripts.evaluate_queries \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="openmask3d_nms_0.5_duplicated" \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.002

python -m openlex3d.visualization.visualize_ranks \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="openmask3d_nms_0.5_duplicated" \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.002

python -m openlex3d.scripts.evaluate_queries \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="concept-graphs" \
masks.alignment_mode="global" \
masks.alignment_threshold=0.01

python -m openlex3d.visualization.visualize_ranks \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="concept-graphs" \
masks.alignment_mode="global" \
masks.alignment_threshold=0.01