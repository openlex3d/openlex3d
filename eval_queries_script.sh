#!/bin/bash

# python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.003 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.003 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.003 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.003 eval.threshold=0.9

# python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.005 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.005 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.005 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.005 eval.threshold=0.9

# python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.01 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.01 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.01 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.01 eval.threshold=0.9

# python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.02 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.02 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.02 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=per_mask masks.alignment_threshold=0.02 eval.threshold=0.9


python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.003 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.003 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.003 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.003 eval.threshold=0.9

python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.005 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.005 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.005 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.005 eval.threshold=0.9

python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.01 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.01 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.01 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.01 eval.threshold=0.9

python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.02 eval.threshold=0.3
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.02 eval.threshold=0.5
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.02 eval.threshold=0.8
python -m openlex3d.scripts.evaluate_queries masks.alignment_mode=global masks.alignment_threshold=0.02 eval.threshold=0.9