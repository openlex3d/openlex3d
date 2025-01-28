import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageColor
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torchmetrics.functional import pairwise_cosine_similarity

from openlex3d.datasets.hm3dsem_walks.create_hm3dsem_walks_gt import (
    PanopticLevel, PanopticObject, PanopticRegion)


class PanopticBuildingEval:
    def __init__(self, building_id):
        self.id = building_id
        self.type = "building"
        self.name = "building"

    def __str__(self):
        return f"{self.id}"


class PanopticLevelEval(PanopticLevel):
    def __str__(self):
        return f"{self.id}"


class PanopticRegionEval(PanopticRegion):
    def __init__(self, region_id, floor_id, category, voted_category, min_height, max_height, mean_height):
        self.id = region_id
        self.floor_id = floor_id
        self.voted_category = voted_category
        self.category = category
        self.hier_id = f"{self.floor_id}_{self.id}"
        self.objects = []
        self.type = "room"

        self.min_height = min_height
        self.max_height = max_height
        self.mean_height = mean_height
        self.region_points = None
        self.bev_region_points = None

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.id}"


class PanopticObjectEval(PanopticObject):
    def __init__(self, object_id, region_id, floor_id, category, hex):
        # semantics object info
        self.id = object_id
        self.hex = hex
        self.category = category
        self.region_id = region_id
        self.floor_id = floor_id
        self.rgb = np.array(ImageColor.getcolor("#" + self.hex, "RGB"))
        self.type = "object"
        self.hier_id = f"{self.floor_id}_{self.region_id}_{self.id}"

        # habitat object info
        self.aabb_center = None
        self.aabb_dims = None
        self.obb_center = None
        self.obb_dims = None
        self.obb_rotation = None
        self.obb_local_to_world = None
        self.obb_world_to_local = None
        self.obb_volume = None
        self.obb_half_extents = None

        # point cloud data
        self.points = None
        self.colors = None

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.region_id}_{self.id}"


class HM3DSemanticEvaluator:
    def __init__(self, params):
        self.params = params
        self.gt_graph = nx.DiGraph()

        self.gt_floors = dict()
        self.gt_rooms = dict()
        self.gt_objects = []

        self.metrics = defaultdict(dict)

    def load_gt_graph(self, path):
        self.gt_scene_infos_path = path
        print("Loading GT graph from: ", self.gt_scene_infos_path)
        with open(self.gt_scene_infos_path, "r") as file:
            scene_info = json.load(file)

        building = PanopticBuildingEval(-1)
        self.gt_graph.add_node(building, name="building", type="building")

        for level_info in scene_info["levels"]:
            level_id = level_info["id"]
            floor = PanopticLevelEval(level_id, level_info["lower"], level_info["upper"])
            floor.regions = level_info["regions"]
            floor.objects = level_info["objects"]

            self.gt_graph.add_node(floor, name=f"floor_{level_id}", type="floor")
            self.gt_graph.add_edge(building, floor)
            self.gt_floors[floor.id] = floor

        for region_info in scene_info["regions"]:
            room = PanopticRegionEval(
                region_info["id"],
                region_info["floor_id"],
                region_info["voted_category"] if "voted_category" in region_info.keys() else None, # region_info["voted_category"],
                region_info["voted_category"],
                region_info["min_height"],
                region_info["max_height"],
                region_info["mean_height"] if "voted_category" in region_info.keys() else None,
            )
            room.graph_id = f"{room.floor_id}_{room.id}"
            room.bev_region_points = np.array(region_info["bev_region_points"])
            room.bev_pcd = o3d.geometry.PointCloud()
            room.bev_pcd.points = o3d.utility.Vector3dVector(room.bev_region_points)
            room.objects = region_info["objects"]

            print(room)

            self.gt_graph.add_node(room, name=f"room_{room.id}", type="room")
            self.gt_graph.add_edge(self.gt_floors[int(room.floor_id)], room)
            self.gt_rooms[room.id] = room

        self.id2obj_idx = {}
        self.hex2obj_idx = {}
        self.hex2id = {}

        idx = 0
        for obj_info in scene_info["objects"]:
            obj = PanopticObjectEval(
                obj_info["id"], obj_info["region_id"], obj_info["floor_id"], obj_info["category"], obj_info["hex"]
            )
            obj.aabb_center, obj.aabb_dims = obj_info["aabb_center"], obj_info["aabb_dims"]
            obj.obb_center, obj.obb_dims = obj_info["obb_center"], obj_info["obb_dims"]
            obj.obb_rotation = obj_info["obb_rotation"]
            obj.obb_local_to_world = obj_info["obb_local_to_world"]
            obj.obb_world_to_local = obj_info["obb_world_to_local"]
            obj.obb_volume = obj_info["obb_volume"]
            obj.obb_half_extents = obj_info["obb_half_extents"]

            # load points from object pcd under self.gt_scene_infos_path + "/objects"
            obj_pcd_path = os.path.join(
                os.path.dirname(self.gt_scene_infos_path), "objects", str(obj_info["id"]) + ".ply"
            )
            obj.pcd = o3d.io.read_point_cloud(obj_pcd_path)
            obj.points = np.asarray(obj.pcd.points)

            self.gt_graph.add_node(obj, name=obj.category, type="object")
            self.gt_graph.add_edge(self.gt_rooms[int(obj.region_id)], obj)
            self.gt_objects.append(obj)


            self.id2obj_idx[obj_info["id"]] = idx
            self.hex2obj_idx[obj_info["hex"]] = idx
            self.hex2id[obj_info["hex"]] = obj_info["id"]
            idx += 1

        self.id2hex = {v: k for k, v in self.hex2id.items()}
        self.id2rgb = {id: ImageColor.getcolor("#" + hex, "RGB") for id, hex in self.id2hex.items()}
        self.rgb2id = {v: k for k, v in self.id2rgb.items()}
        self.obj_idx2id = {v: k for k, v in self.id2obj_idx.items()}

        print("----------------------------")
        print("GT graph loaded:")
        print("Number of GT floors: ", len([node for node in self.gt_graph.nodes if node.type == "floor"]))
        print("Number of GT rooms: ", len([node for node in self.gt_graph.nodes if node.type == "room"]))
        print("Number of GT objects: ", len([node for node in self.gt_graph.nodes if node.type == "object"]))
        print("----------------------------")

    def get_panoptic_pointclouds(self, dataset, frame_idx, filt_dist, binary_masks=False):
        _, depth_path, pose_path, panoptic_path = dataset.data_list[frame_idx]
        panoptic_ids = np.load(panoptic_path)
        depth = np.array(dataset._load_depth(depth_path))
        pose = dataset._load_pose(pose_path)

        id2mask = {}
        id2binary_mask = {}
        # panoptic_array = panoptic_ids # np.array(panoptic_image)
        # panoptic_rgb_values = np.unique(panoptic_array.reshape(-1, panoptic_array.shape[2]), axis=0)

        panoptic_rgb = np.zeros((panoptic_ids.shape[0], panoptic_ids.shape[1], 3), dtype=np.uint8)
        for id, id_rgb in self.id2rgb.items():
            panoptic_rgb[panoptic_ids == id, :] = id_rgb
        panoptic_image = Image.fromarray(panoptic_rgb) # as sanity check
        panoptic_dir_path = os.path.join(self.params.paths.data,
                                         self.params.main.dataset,
                                         self.params.main.split,
                                         self.params.main.scene,
                                         "panoptic")
        if not os.path.exists(panoptic_dir_path):
            os.makedirs(panoptic_dir_path)
        panoptic_image.save(os.path.join(panoptic_dir_path, self.params.main.scene.split("-")[1] + "_" + str(f"{frame_idx:06}") + ".png"))
        panoptic_array = np.array(panoptic_image)
        panoptic_rgb_values = np.unique(panoptic_array.reshape(-1, panoptic_array.shape[2]), axis=0)

        for _, rgb_tuple in enumerate(panoptic_rgb_values):
            if np.sum(rgb_tuple) == 0:
                continue
            mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=bool)
            mask_indices = np.where(np.all(panoptic_array == rgb_tuple, axis=2))
            # disregard small masks (smaller than approx. 1% of the panoptic image)
            if len(mask_indices[0]) < 10000:
                continue
            mask[mask_indices] = True
            mask_pointcloud = dataset.get_pointcloud(panoptic_image, depth, cam2global=pose, mask=mask, filter_distance=filt_dist)
            id2mask[self.rgb2id[tuple(rgb_tuple)]] = mask_pointcloud
            id2binary_mask[self.rgb2id[tuple(rgb_tuple)]] = mask

        if binary_masks:
            return id2mask, id2binary_mask
        else:
            return id2mask