import numpy as np
import json

def calculated_IOU(eval_segm, gt_segm, labels, object_class_mapping, mappings):
    intersection = np.zeros(len(mappings))
    union = np.zeros(len(mappings))

    mapping_labels = []
    
    for eval_point, gt_label in zip(eval_segm, gt_segm):
        eval_label = labels[int(eval_point)]
        gt_id = gt_label[0]
        
        if gt_id == 0:
            mapping_labels.append("none")
            continue
        
        gt_class_info = object_class_mapping.get(gt_id)
        if len(gt_class_info["synonyms"]) == 0:
            mapping_labels.append("none")
            continue
        
        cl = gt_class_info["class_name"]
        if cl in ["wall", "floor", "ceiling", "window", "door", "rug", "undefined", "switch", "pillar", "wall-plug"]:
            mapping_labels.append("none")
            continue

        level = "incorrect"
        for i, mapping in enumerate(mappings):
            mapped_labels = gt_class_info.get(mapping, [])
            union[i] += 1
            if eval_label in mapped_labels:
                intersection[i] += 1
                level = mapping
                break
            
        mapping_labels.append(level)
    
    ious = [intersection[i] / union[i] if union[i] != 0 else 0 for i in range(len(mappings))]
    return ious, mapping_labels

def mean_IOU(eval_segm, gt_segm, labels, semantic_info_path, mappings=["synonyms", "vis_sim", "related"], ignore=[]):
    """
    Calculate mean Intersection over Union (IoU) for each mapping type.
    
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param labels: list or numpy array, label information
    :param semantic_info_path: str, path to semantic information JSON file
    :param mappings: list of str, mappings to calculate IoU for (default: ['synonyms', 'vis_sim', 'related'])
    :param ignore: list of classes to ignore
    :return: list of IoU values for each mapping
    """
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    object_class_mapping = {
        obj["id"]: {
            "class_id": obj["class_id"],
            "class_name": obj["class_name"],
            "synonyms": obj["synonyms"],
            "vis_sim": obj["vis_sim"],
            "related": obj["related"]
        }
        for obj in semantic_info.get("objects", [])
    }

    ious, mapping_labels = calculated_IOU(eval_segm, gt_segm, labels, object_class_mapping, mappings)
    
    return ious, mapping_labels


