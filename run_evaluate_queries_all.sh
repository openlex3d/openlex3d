#!/bin/bash

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs" \
eval.metric="rank" \
eval.top_k=20

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs" \
eval.metric="rank" \
eval.top_k=20

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs" \
eval.metric="ap"

python -m openlex3d.visualization.visualize_prec_recall_curve \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs" \
eval.metric="ap"

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated" \
eval.metric="rank" \
eval.top_k=20 \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.001

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated" \
eval.metric="rank" \
eval.top_k=20 \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.001

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated" \
eval.top_k=20 \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.001

python -m openlex3d.visualization.visualize_prec_recall_curve \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated" \
eval.top_k=20 \
masks.alignment_mode="per_mask" \
masks.alignment_threshold=0.001