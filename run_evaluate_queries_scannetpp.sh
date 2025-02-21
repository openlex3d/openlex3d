#!/bin/bash

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_nms_0.5_duplicated"

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_nms_0.5_duplicated"

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_nms_0.75_duplicated"

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_nms_0.75_duplicated"

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated"

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="openmask3d_duplicated"

python -m openlex3d.scripts.evaluate_queries \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs"

python -m openlex3d.visualization.visualize_ranks \
dataset.name="scannetpp" \
dataset.scenes='["0a76e06478", "1f7cbbdde1", "49a82360aa", "c0f5742640", "fd361ab85f", "0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"]' \
dataset.path="/home/kumaraditya/datasets/scannetpp_openlex_v2" \
pred.method="concept-graphs"