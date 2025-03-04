import numpy as np
from openlex3d.models import load_model
from openlex3d.core.evaluation import compute_feature_to_prompt_similarity


def compute_normalized_cosine_similarities(model_cfg, pred_features, query_texts):
    clip_model = load_model(model_cfg)
    sim_matrix = compute_feature_to_prompt_similarity(
        clip_model, pred_features, query_texts
    )  # (n_instances, n_queries)

    # Normalize from [-1,1] to [0,1]
    norm_sim = (sim_matrix + 1) / 2.0

    # Compute min-max normalized similarity for each query
    min_vals = norm_sim.min(axis=0, keepdims=True)  # Shape: (1, n_queries)
    max_vals = norm_sim.max(axis=0, keepdims=True)  # Shape: (1, n_queries)
    range_vals = np.maximum(max_vals - min_vals, 1e-8)
    min_max_norm_sim = (norm_sim - min_vals) / range_vals

    # print(query_texts[1])
    # print(min_max_norm_sim[:, 1])

    return min_max_norm_sim
