#!/usr/bin/env python
# -*- coding: UTF8 -*-
# PYTHON_ARGCOMPLETE_OK

import logging
import hydra
import numpy as np

from omegaconf import DictConfig

from openlex3d import get_path
from openlex3d.datasets import load_dataset
from openlex3d.models import load_model
from openlex3d.core.evaluation import (
    compute_feature_to_prompt_similarity,
    get_label_from_logits,
)
from openlex3d.core.io import (
    load_predicted_features,
    load_prompt_list,
)
from openlex3d.core.metric import intersection_over_union  # noqa


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=f"{get_path()}/config", config_name="replica"
)
def main(config: DictConfig):
    # Load dataset
    gt_cloud, gt_labels = load_dataset(config.dataset)

    # Run evaluation
    if config.evaluation.type == "features":
        # Load language model
        model = load_model(config.model)

        # Load predicted features
        pred_cloud, pred_feats = load_predicted_features(
            config.evaluation.predictions_path,
            config.evaluation.voxel_downsampling_size,
        )

        # Load prompt list
        prompt_list = load_prompt_list(config.dataset.openlex3d_path)

        # Evaluate predicted features
        logits = compute_feature_to_prompt_similarity(
            model=model,
            features=pred_feats,
            prompt_list=prompt_list,
        )

        # Get predicted label from logits
        pred_labels = get_label_from_logits(logits, prompt_list)

        # Concatenate point coordinates and labels for predicted and ground truth cloud
        pred_coords = np.zeros((len(pred_cloud.point.positions), 4))
        pred_coords[:, :3] = pred_cloud.point.positions.numpy()
        pred_coords[:, -1] = pred_labels

        gt_coords = np.zeros((len(gt_cloud.point.positions), 4))
        gt_coords[:, :3] = gt_cloud.point.positions.numpy()
        gt_coords[:, -1] = gt_labels

        # ious, accs, mapping_labels = metric.IOU(
        #         pred_coords,
        #         gt_coords,
        #         prompt_list,
        #         semantic_info_path,
        #         dataset=params.main.dataset,
        #         typ=typ,
        #     )

    elif config.evaluation.type == "caption":
        raise NotImplementedError(f"{config.evaluation.type} not supported yet")
    else:
        raise NotImplementedError(f"{config.evaluation.type} not supported")


if __name__ == "__main__":
    main()
