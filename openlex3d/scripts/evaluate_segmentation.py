#!/usr/bin/env python
# -*- coding: UTF8 -*-
# PYTHON_ARGCOMPLETE_OK

import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path

import openlex3d.core.metric as metric  # noqa
from openlex3d import get_path
from openlex3d.core.evaluation import (
    compute_feature_to_prompt_similarity,
    get_label_from_logits,
)
from openlex3d.core.io import load_predicted_features, load_prompt_list, save_results
from openlex3d.datasets import load_dataset
from openlex3d.models import load_model

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=f"{get_path()}/config", config_name="replica"
)
def main(config: DictConfig):
    # Load dataset
    gt_cloud, gt_ids, openlex3d_gt_handler = load_dataset(
        config.dataset, load_openlex3d=True
    )

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
        prompt_list = load_prompt_list(config)

        # Evaluate predicted features
        logits = compute_feature_to_prompt_similarity(
            model=model,
            features=pred_feats,
            prompt_list=prompt_list,
        )

        # Get predicted label from logits
        pred_labels = get_label_from_logits(
            logits, prompt_list, method="topn", topn=config.evaluation.topn
        )

        results = {}
        pred_categories = None
        point_labels = None
        point_categories = None
        # Compute metric (intersection over union)
        if config.evaluation.freq:
            freq_results, pred_categories, point_labels, point_categories = (
                metric.category_frequency_topn(
                    pred_cloud=pred_cloud,
                    pred_labels=pred_labels,
                    gt_cloud=gt_cloud,
                    gt_ids=gt_ids,
                    gt_labels_handler=openlex3d_gt_handler,
                    excluded_labels=config.evaluation.excluded_labels,
                )
            )
            results["freq"] = freq_results

        if config.evaluation.set_ranking:
            set_ranking_results = metric.set_based_ranking(
                pred_cloud=pred_cloud,
                gt_cloud=gt_cloud,
                gt_ids=gt_ids,
                gt_labels_handler=openlex3d_gt_handler,
                excluded_labels=config.evaluation.excluded_labels,
                logits=logits,
                prompt_list=prompt_list,
            )
            results["ranking"] = set_ranking_results

        # Log results
        logger.info(f"Scene Metrics: {results}")

        # Export predicted clouds
        save_results(
            output_path=config.evaluation.output_path,
            dataset=config.dataset.name,
            scene=config.dataset.scene,
            algorithm=Path(
                config.evaluation.algorithm, f"top_{config.evaluation.topn}"
            ),
            reference_cloud=gt_cloud,
            pred_categories=pred_categories,
            results=results,
            point_labels=point_labels,
            point_categories=point_categories,
        )
        # log results saved to
        logger.info(
            f"Results saved to {config.evaluation.output_path}"
            f"/{config.evaluation.algorithm}/top_{config.evaluation.topn}"
            f"/{config.dataset.name}/{config.dataset.scene}"
        )

    elif config.evaluation.type == "caption":
        raise NotImplementedError(f"{config.evaluation.type} not supported yet")
    else:
        raise NotImplementedError(f"{config.evaluation.type} not supported")


if __name__ == "__main__":
    main()
