#!/bin/bash

python -m openlex3d.scripts.evaluate_queries \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="openmask3d_nms_0.5_duplicated"

python -m openlex3d.visualization.visualize_ranks \
dataset.name="replica" \
dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
dataset.path="/home/kumaraditya/datasets/Replica_original" \
pred.method="openmask3d_nms_0.5_duplicated"

# python -m openlex3d.visualization.visualize_ranks \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="openmask3d_duplicated" \
# eval.metric="rank" \
# eval.top_k=20

# python -m openlex3d.scripts.evaluate_queries \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="concept-graphs" \
# eval.metric="rank" \
# eval.top_k=20

# python -m openlex3d.visualization.visualize_ranks \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="concept-graphs" \
# eval.metric="rank" \
# eval.top_k=20

# python -m openlex3d.scripts.evaluate_queries \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="concept-graphs" \
# eval.metric="ap"

# python -m openlex3d.visualization.visualize_prec_recall_curve \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="concept-graphs" \
# eval.metric="ap"

# python -m openlex3d.scripts.evaluate_queries \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="openmask3d_duplicated" \
# eval.metric="rank" \
# eval.top_k=20

# python -m openlex3d.visualization.visualize_ranks \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="openmask3d_duplicated" \
# eval.metric="rank" \
# eval.top_k=20

# python -m openlex3d.scripts.evaluate_queries \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="openmask3d_duplicated" \
# eval.metric="ap"

# python -m openlex3d.visualization.visualize_prec_recall_curve \
# dataset.name="replica" \
# dataset.scenes='["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]' \
# dataset.path="/home/kumaraditya/datasets/Replica_original" \
# pred.method="openmask3d_duplicated" \
# eval.metric="ap"