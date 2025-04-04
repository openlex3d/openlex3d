import open3d as o3d
import numpy as np
import plyfile
import json

from pathlib import Path


def load_dataset(name: str, scene: str, base_path: str):
    # Read original ground truth PLY
    # Prepare input paths
    scene = "_".join([scene[:-1], scene[-1]])
    dataset_root = Path(base_path, scene)

    semantic_info_path = dataset_root / "habitat" / "info_semantic.json"
    assert semantic_info_path.exists()

    ply_path = dataset_root / "habitat" / "mesh_semantic.ply"
    assert ply_path.exists()

    # Load cloud
    gt_cloud, gt_labels = read_ply(str(ply_path), str(semantic_info_path))
    assert gt_cloud.point.positions.shape[0] > 0
    assert len(gt_labels) > 0

    return gt_cloud, gt_labels


# Function to read PLY file and assign colors based on object_id for replica dataset
def read_ply(file_path: str, semantic_info_path: str):
    """
    Read PLY file and assign colors based on object_id for replica dataset
    :param file_path: path to PLY file
    :param semantic_info_path: path to semantic info JSON file
    :return: point cloud, class ids, point cloud instance, object ids
    """
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    object_class_mapping = {
        obj["id"]: {
            "class_id": obj["class_id"],
        }
        for obj in semantic_info["objects"]
    }

    unique_class_ids = {obj["class_id"] for obj in semantic_info["objects"]}
    unique_class_ids = np.array(list(unique_class_ids))

    # Extract vertex data
    vertices = np.vstack(
        [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]
    ).T

    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)

    # set random color for every unique object_id/instance id
    unique_object_ids = np.unique(object_ids)
    instance_colors = np.zeros((len(object_ids1), 3))
    unique_colors = np.random.rand(len(unique_object_ids), 3)
    for i, object_id in enumerate(unique_object_ids):
        instance_colors[object_ids1 == object_id] = unique_colors[i]

    # semantic colors
    class_ids = []
    for object_id in object_ids1:
        if object_id in object_class_mapping.keys():
            # class_ids.append(object_class_mapping[object_id])
            class_ids.append(object_id)
        else:
            class_ids.append(0)
    class_ids = np.array(class_ids)

    class_colors = np.zeros((len(object_ids1), 3))
    unique_class_colors = np.random.rand(len(unique_class_ids), 3)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[class_ids == class_id] = unique_class_colors[i]

    # Make point cloud
    cloud = o3d.t.geometry.PointCloud(vertices1)
    cloud.point.colors = o3d.core.Tensor(class_colors)

    return cloud, class_ids


def load_dataset_with_obj_ids(name: str, scene: str, base_path: str):
    # Read original ground truth PLY
    # Prepare input paths

    # change scene name by adding a _ between the word and number, for eg: office4 to office_4
    scene = "_".join([scene[:-1], scene[-1]])
    dataset_root = Path(base_path, scene)

    semantic_info_path = dataset_root / "habitat" / "info_semantic.json"
    # print(semantic_info_path)
    assert semantic_info_path.exists()

    ply_path = dataset_root / "habitat" / "mesh_semantic.ply"
    assert ply_path.exists()

    # Load cloud
    gt_cloud, obj_ids = read_ply_with_obj_ids(str(ply_path), str(semantic_info_path))
    assert gt_cloud.point.positions.shape[0] > 0
    assert obj_ids.shape[0] > 0

    return gt_cloud, obj_ids


def read_ply_with_obj_ids(file_path: str, semantic_info_path: str):
    """
    Read PLY file and assign colors based on object_id for replica dataset
    :param file_path: path to PLY file
    :param semantic_info_path: path to semantic info JSON file
    :return: point cloud, class ids, point cloud instance, object ids
    """
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    unique_class_ids = {obj["class_id"] for obj in semantic_info["objects"]}
    unique_class_ids = np.array(list(unique_class_ids))

    # Extract vertex data
    vertices = np.vstack(
        [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]
    ).T

    colors = (
        np.vstack(
            [
                plydata["vertex"]["red"],
                plydata["vertex"]["green"],
                plydata["vertex"]["blue"],
            ]
        ).T.astype(np.float32)
        / 255.0
    )

    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    colors1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
        colors1.append(colors[face])
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)
    colors1 = np.vstack(colors1)

    cloud = o3d.t.geometry.PointCloud(vertices1)
    cloud.point.colors = o3d.core.Tensor(colors1)

    return cloud, object_ids1


if __name__ == "__main__":
    pass
