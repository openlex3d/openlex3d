# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
import copy
import gc
import os
from collections import defaultdict
from typing import Any, List

import cv2
import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import torch
import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.distance import cdist

gc.collect()


from openlex3d.datasets.hm3dsem import HM3DSemanticEvaluator
from openlex3d.datasets.hm3dsem_walks.hm3dsem_walks import HM3DSemWalksDataset


@hydra.main(version_base=None, config_path="../../../config", config_name="synonym")
def main(params: DictConfig):
    # scenes = [
    #         # "00824-Dd4bFSTQ8gi",
    #         "00829-QaLdnwvtxbs"]
    scenes = params.training.val_scenes
    generate(params, scenes, None)


def generate(params: DictConfig, scenes: list, graph_db_dir: str):

    for scene in scenes:
        # Load depth, RGB, pose, and intrinsics
        HM3DSemWalksDataset
        observations = HM3DSemWalksDataset({"data_dir": params.paths.data, "transforms": None})

        # TODO Load ground-truth objects from dataset
        gt_graph = HM3DSemanticEvaluator(params=params)
        gt_graph.load_gt_graph(os.path.join(params.paths.data,
                                            params.main.dataset,
                                            params.main.split,
                                            scene,
                                            "scene_info.json"))

        id2frames = defaultdict(list)
        id_positions = defaultdict(list)

        # Sweeping through all camera poses of the walk
        stride = 5
        for idx in tqdm.tqdm(range(0, int(len(observations)), stride), desc="Go through camera poses"):
            _rgb_image, _depth_image, _pose, _, intrinsics = observations[idx]
            rgb_path, depth_path, pose_path, _ = observations.get_paths(idx)

            # Extract frame-respective ground truth
            id2panoptic_masks = gt_graph.get_panoptic_pointclouds(observations, idx, params.mapping.max_mask_distance)
            # id2mask[self.rgb2id[tuple(rgb_tuple)]] = mask_pointcloud

            gt_graph.id2rgb = {v: k for k,v in gt_graph.rgb2id.items()}

            for id, panoptic_pcd in tqdm.tqdm(id2panoptic_masks.items(), "Load frame-wise GT objects"):

                id_rgb_image = copy.copy(_rgb_image)
                panoptic_array = np.load(observations.data_list[idx][-1])[:, :, np.newaxis]

                id_rgb_tuple = gt_graph.id2rgb[id]
                mask_indices = np.where(np.all(panoptic_array == id, axis=2))

                mask = np.zeros((panoptic_array.shape[0], panoptic_array.shape[1]), dtype=bool)
                mask[mask_indices] = True

                assert np.sum(mask) > 100, f"Mask is np.empty for ID {id}, frame {frame_idx}"

                id2frames[id].append([idx, len(panoptic_pcd.points), _pose[0:3, 2].tolist()])

                # draw mask on top of rgb
                rgb_array = np.array(id_rgb_image)
                rgb_array[mask] = [255, 0, 0]
                rgb_pil = Image.fromarray(rgb_array)
                mask_save_path = os.path.join(params.main.results_path, scene, "masks", f"{id}")
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                rgb_pil.save(os.path.join(mask_save_path, f"{idx:06}.png"))


        id2sel_frames = defaultdict(list)

        for id, list_of_frames in id2frames.items():
            selected_frames = list()

            list_of_frames.sort(key=lambda x: x[1], reverse=True)
            pos_array = np.array([frame[2] for frame in list_of_frames])
            if pos_array.shape[0] < 9:
                # select all frames available
                selected_frames = list_of_frames
                id2sel_frames[id] = selected_frames
            else:
                num = 1
                taken_indices = [0]
                # take best frame first and then do farthest-point sampling
                best_frame_pos = np.array(list_of_frames[0][2])[np.newaxis, :]

                while num < 9:
                    dist_to_curr_best = cdist(best_frame_pos, pos_array).flatten()
                    farthest_idcs = np.argsort(dist_to_curr_best)
                    # Go through list of idcs and take the first one that is not contained in taken_indices
                    for idx in farthest_idcs:
                        if idx not in taken_indices:
                            taken_indices.append(idx)
                            num += 1
                            best_frame_pos = pos_array[idx][np.newaxis, :]
                            break

                # convert relative indices to absolute frame indices
                for taken_idx in taken_indices:
                    selected_frames.append(list_of_frames[taken_idx])
                id2sel_frames[id] = selected_frames


            fig = plt.figure(figsize=(16., 6.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(2, 4),  # creates 2x2 grid of Axes
                            axes_pad=0.1,  # pad between Axes in inch.
                            )

            for ax, sample in zip(grid, selected_frames):

                frame_idx = sample[0]
                _rgb_image, _depth_image, _pose, _, intrinsics = observations[frame_idx]
                panoptic_array = np.load(observations.data_list[frame_idx][-1])[:, :, np.newaxis]

                id_rgb_tuple = gt_graph.id2rgb[id]
                mask_indices = np.where(np.all(panoptic_array == id, axis=2))

                mask = np.zeros((panoptic_array.shape[0], panoptic_array.shape[1]), dtype=bool)
                mask[mask_indices] = True

                rgb_array = np.array(_rgb_image)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_array, contours, -1, (255, 0, 0), 4)

                mask_color = np.zeros_like(rgb_array)
                mask_color[mask == True] = (255, 0, 0)

                alpha = 0.1
                blended_rgb_array = cv2.addWeighted(mask_color, alpha, rgb_array, 1 - alpha, 0)

                ax.imshow(blended_rgb_array)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            if len(selected_frames) < grid.ngrids:
                for ax in grid[len(selected_frames):]:
                    ax.axis('off')
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
            plt.tight_layout()
            plt.suptitle(f"{scene} - ID {id:06}", fontweight='bold', font='Arial')
            plt.subplots_adjust(top=0.95)
            plt.savefig(os.path.join(params.main.results_path, scene, f"{id:06}.png"))
            plt.clf()

        print("Went through all frames")


if __name__ == "__main__":
    main()