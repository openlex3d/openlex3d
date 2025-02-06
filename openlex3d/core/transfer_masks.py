import numpy as np
import faiss


def get_pred_mask_indices_gt_aligned_global(
    pred_pcd, pred_mask_indices, mesh_vertices, threshold
):
    """
    Global mode:
    Build a single FAISS index on all predicted pcd points.
    For each mesh vertex, query its nearest predicted point.
    If the distance is below 'threshold', assign that vertex to the predicted instance.
    Return a dictionary mapping predicted instance id -> numpy array of mesh vertex indices.
    """
    d = pred_pcd.shape[1]
    index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(np.ascontiguousarray(pred_pcd.astype("float32")))
    # This returns one nearest neighbor for each query point (i.e. each mesh vertex).
    distances, indices = gpu_index.search(
        np.ascontiguousarray(mesh_vertices.astype("float32")), 1
    )
    distances = distances.ravel()
    nearest_pred_indices = indices.ravel()  # indices into pred_pcd
    aligned_dict = {}
    for i, d_val in enumerate(distances):
        if d_val < threshold:
            instance_id = int(pred_mask_indices[nearest_pred_indices[i]])
            aligned_dict.setdefault(instance_id, []).append(i)
    # Convert lists to numpy arrays.
    for instance_id in aligned_dict:
        aligned_dict[instance_id] = np.array(aligned_dict[instance_id])
    return aligned_dict


def get_pred_mask_indices_gt_aligned_per_mask(
    pred_pcd, pred_mask_indices, mesh_vertices, threshold
):
    """
    Per-mask mode:
    For each unique predicted instance id, build a FAISS index on only its predicted pcd points.
    Then query all mesh vertices and assign those for which the nearest neighbor distance is less than 'threshold'.
    Return a dictionary mapping predicted instance id -> numpy array of mesh vertex indices.
    """
    aligned_dict = {}
    unique_instances = np.unique(pred_mask_indices)
    d = pred_pcd.shape[1]
    for instance_id in unique_instances:
        mask = pred_mask_indices == instance_id
        if np.sum(mask) == 0:
            continue
        instance_points = pred_pcd[mask]
        index = faiss.IndexFlatL2(d)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(np.ascontiguousarray(instance_points.astype("float32")))
        distances, _ = gpu_index.search(
            np.ascontiguousarray(mesh_vertices.astype("float32")), 1
        )
        distances = distances.ravel()
        indices_assigned = np.nonzero(distances < threshold)[0]
        if len(indices_assigned) > 0:
            aligned_dict[int(instance_id)] = indices_assigned
    return aligned_dict
