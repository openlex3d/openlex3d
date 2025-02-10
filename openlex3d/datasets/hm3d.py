import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import open3d as o3d
import torch
from PIL import ImageColor
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity


class PanopticBuildingEval:
    def __init__(self, building_id):
        self.id = building_id
        self.type = "building"
        self.name = "building"

    def __str__(self):
        return f"{self.id}"


class PanopticLevelEval:
    def __init__(self, level_id, lower, upper):
        self.id = level_id
        self.lower = lower
        self.upper = upper
        self.type = "floor"

        self.regions = []
        self.objects = []

    def __str__(self):
        return f"{self.id}"
    
    def __print__(self) -> str:
        return f"{self.id}"


class PanopticRegionEval():
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


class PanopticObjectEval():
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

        self.synonyms = []
        self.vis_sim = []
        self.depictions = []
        self.clutter = []

    def __str__(self) -> str:
        if any([len(self.synonyms) > 0, len(self.vis_sim) > 0, len(self.depictions) > 0]):
            return f"obj_id: {self.id}, category: {self.category}, region_id: {self.region_id}, synonyms: {self.synonyms}, vis_sim: {self.vis_sim}, depictions: {self.depictions}"
        else: 
            return f"obj_id: {self.id}, category: {self.category}, region_id: {self.region_id}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_colored_pcd(self):
        colored_pcd = copy.copy(self.pcd)
        colored_pcd.colors = o3d.utility.Vector3dVector(np.tile(self.rgb, (len(self.points), 1)) / 255.0)
        return colored_pcd
    


class HM3DSemanticGT:
    def __init__(self):
        self.gt_graph = nx.DiGraph()

        self.gt_floors = dict()
        self.gt_rooms = dict()
        self.gt_objects = dict()

        self.panoptic_pcd = None

    def load_gt_graph_from_json(self, path):
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
                region_info["category"],
                region_info["voted_category"],
                region_info["min_height"],
                region_info["max_height"],
                region_info["mean_height"],
            )
            room.graph_id = f"{room.floor_id}_{room.id}"
            room.bev_region_points = np.array(region_info["bev_region_points"])
            room.bev_pcd = o3d.geometry.PointCloud()
            room.bev_pcd.points = o3d.utility.Vector3dVector(room.bev_region_points)
            room.objects = region_info["objects"]

            self.gt_graph.add_node(room, name=f"room_{room.id}", type="room")
            self.gt_graph.add_edge(self.gt_floors[int(room.floor_id)], room)
            self.gt_rooms[room.id] = room

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
            self.gt_objects[obj.id] = obj


        print("----------------------------")
        print("GT graph loaded:")
        print("Number of GT floors: ", len([node for node in self.gt_graph.nodes if node.type == "floor"]))
        print("Number of GT rooms: ", len([node for node in self.gt_graph.nodes if node.type == "room"]))
        print("Number of GT objects: ", len([node for node in self.gt_graph.nodes if node.type == "object"]))
        print("----------------------------")


    def add_synoym_labels(self, syn_path):
        self.synonym_labels_path = syn_path
        print("Loading GT graph from: ", self.synonym_labels_path)
        with open(self.synonym_labels_path, "r") as file:
            synonym_labels = json.load(file)

        for syn_obj in synonym_labels["dataset"]["samples"]:
            self.gt_objects[syn_obj["object_id"]].synonyms = syn_obj["labels"]["image_attributes"]["synonyms"]
            self.gt_objects[syn_obj["object_id"]].vis_sim = syn_obj["labels"]["image_attributes"]["vis_sim"]
            self.gt_objects[syn_obj["object_id"]].depictions = syn_obj["labels"]["image_attributes"]["depictions"]
    

    def object_semantics_eval_tp_auc(
        self, top_k_spec, row_ind, col_ind, pred_objects, gt_objects, gt_text_feats, gt_classes
    ):
        success_k = {k: list() for k in top_k_spec}
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            # dot_sim = np.dot(pred_objects[pred_idx].embedding, gt_text_feats.T)
            dot_sim = (
                pairwise_cosine_similarity(
                    torch.from_numpy(pred_objects[pred_idx].embedding.reshape(1, -1)).float(),
                    torch.from_numpy(gt_text_feats).float(),
                )
                .squeeze(0)
                .numpy()
            )
            # sort the dot similarity scores in descending order
            sorted_dot_similarity = np.sort(dot_sim)[::-1]
            for k in top_k_spec:
                top_k_idx = np.argsort(dot_sim)[::-1][:k]
                # get names of top k classes
                top_k_classes = [gt_classes[idx] for idx in top_k_idx]
                if gt_objects[gt_idx].category in top_k_classes:
                    success_k[k].append((pred_idx))
        top_k_acc = {k: len(v) / len(col_ind) for k, v in success_k.items()}

        norm_top_k = [k / len(gt_classes) for k in top_k_spec]
        tp_top_k_auc = np.trapz(list(top_k_acc.values()), norm_top_k)
        return top_k_acc, tp_top_k_auc
    

    def get_rgb_pointcloud(self) -> o3d.geometry.PointCloud:
        """
        Returns a point cloud with all objects colored by their original RGB values
        
        Returns:
            o3d.geometry.PointCloud: Point cloud with all objects colored by their original RGB values
            Dict[int, PanopticObjectEval]: Dictionary of all objects with object id as key
        """

        assert len(self.gt_objects) > 0, "GT objects not loaded"
        self.rgb_pcd = o3d.geometry.PointCloud()
        for obj_id, obj in self.gt_objects.items():
            self.rgb_pcd += obj.pcd
        return self.rgb_pcd, self.gt_objects
    

    def get_panoptic_pointcloud(self) -> o3d.geometry.PointCloud:
        """
        Returns a point cloud with all objects colored by their respective instance RGB values.
        
        Returns:
            o3d.geometry.PointCloud: Point cloud with all objects colored by their original RGB values
            Dict[int, PanopticObjectEval]: Dictionary of all objects with object id as key
        """

        return self.panoptic_pcd, self.gt_objects
    

def load_dataset(name: str = None, scene: str = None, base_path: str = None) -> tuple[o3d.t.geometry.PointCloud, np.ndarray]:
    """
    Load the HM3DSemantics-walks ground truth point cloud and instance labels

    Args:
        name (str): Dataset name
        scene (str): Scene ID
        base_path (str): Base path to the dataset

    Returns:
        tuple[o3d.t.geometry.PointCloud, np.ndarray]: Ground truth point cloud and instance labels

    """

    assert name, "dataset name not defined, please provide a dataset name"
    assert scene, "scene not defined, please provide a given scene identifier"
    assert base_path, "dataset.path not defined, please add a path to the HM3DSemantics-walks dataset"

    dataset_root = Path(base_path, f"{scene}")

    gt_scene_info_path = dataset_root / "scene_info.json"
    assert gt_scene_info_path.exists()

    hm3d_gt = HM3DSemanticGT()
    hm3d_gt.load_gt_graph_from_json(gt_scene_info_path)
    assert len(hm3d_gt.gt_objects) > 0, "GT objects not loaded"

    gt_instance_labels = list() 
    points, colors = list(), list()

    for obj_id, obj in hm3d_gt.gt_objects.items():
        labels = np.full(len(obj.pcd.points), obj_id, dtype=np.uint16)
        gt_instance_labels.append(labels)
        points.append(np.asarray(obj.pcd.points))
        colors.append(np.asarray(obj.pcd.colors))

    gt_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.concatenate(points)))
    gt_cloud.point.colors = o3d.core.Tensor(np.concatenate(colors))
    gt_instance_labels = np.concatenate(gt_instance_labels)
    assert gt_instance_labels.shape[0] == len(gt_cloud.point.positions), "Length of GT instance labels does not match number of cloud points"

    return gt_cloud, gt_instance_labels

if __name__ == "__main__":

    cloud, labels = load_dataset("hm3dsem_walks", "00824-Dd4bFSTQ8gi", "/home/buechner/data/hm3dsem_walks/val")
    o3d.visualization.draw([cloud])

    # use of HM3DSem GT class for reference
    "/home/buechner/data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi/scene_info.json", 
    "/home/buechner/ovsg-dataset/merged_labels/hm3d/00824/hm3dsem_00824_merged_reviewed.json"  
    
    # gt_path = sys.argv[1] # provide hm3dsem_walks GT JSON path
    # syn_path = sys.argv[2] # read in synonym labels

    # hm3d_gt = HM3DSemanticGT()
    # hm3d_gt.load_gt_graph_from_json(gt_path)
    # hm3d_gt.add_synoym_labels(syn_path)

    # # print all objects
    # for id, obj in hm3d_gt.gt_objects.items():
    #     print(obj)

    # # get rgb point cloud
    # rgb_pcd = hm3d_gt.get_gt_pointcloud()
    # o3d.visualization.draw_geometries([rgb_pcd])

    # # get panoptic point cloud
    # panoptic_pcd = hm3d_gt.get_panoptic_pointcloud()
    # o3d.visualization.draw_geometries([panoptic_pcd])

