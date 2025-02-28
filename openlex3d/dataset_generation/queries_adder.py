import json
from pathlib import Path

EXCLUDED_LABELS = ["wall", "floor", "ceiling", "doorframe", "ledge", "windowledge"]


def add_queries_to_scene(data):
    for sample in data["dataset"]["samples"]:
        labels = sample.get("labels", {})
        synonyms = labels.get("image_attributes", {}).get("synonyms", [])
        depictions = labels.get("image_attributes", {}).get("depictions", [])

        # Build level0 queries
        level0 = synonyms

        # Build level1 queries
        if depictions and synonyms:
            level1 = [
                f"{depiction} {synonym}"
                for depiction in depictions
                for synonym in synonyms
            ]
        else:
            level1 = []

        # Add queries to the sample
        sample["queries"] = {"level0": level0, "level1": level1}

    return data


def build_query_to_obj_mapping(data):
    query_mapping = {"level0": {}, "level1": {}}

    for sample in data["dataset"]["samples"]:
        obj_id = sample["object_id"]
        queries = sample.get("queries", {})

        for level, queries_list in queries.items():
            for query in queries_list:
                query_space_removed = query.replace(" ", "")
                if query_space_removed in EXCLUDED_LABELS:
                    continue

                if query not in query_mapping[level]:
                    query_mapping[level][query] = []
                query_mapping[level][query].append(obj_id)

    # del query_mapping["level1"]

    return query_mapping


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def process_scene_labels(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    base_name = Path(input_path).stem
    base_path = Path(input_path).parent
    scene_queries_output_path = base_path / f"{base_name}_w_queries.json"
    query_mapping_all_output_path = base_path / f"{base_name}_query_to_object_mapping_all.json"
    query_mapping_l0_output_path = base_path / f"{base_name}_query_to_object_mapping_l0.json"
    query_mapping_l1_output_path = base_path / f"{base_name}_query_to_object_mapping_l1.json"

    updated_scene_data = add_queries_to_scene(data)

    query_mapping = build_query_to_obj_mapping(updated_scene_data)

    # Separate level0 and level1 mappings
    query_mapping_l0 = {"level0": query_mapping["level0"]}
    query_mapping_l1 = {"level1": query_mapping["level1"]}

    save_json(updated_scene_data, scene_queries_output_path)
    save_json(query_mapping, query_mapping_all_output_path)
    save_json(query_mapping_l0, query_mapping_l0_output_path)
    save_json(query_mapping_l1, query_mapping_l1_output_path)

    return


def process_openlex_labels(openlex_labels_folder):
    openlex_labels_folder = Path(openlex_labels_folder)
    dataset_folders = [f for f in openlex_labels_folder.iterdir() if f.is_dir()]

    for dataset_folder in dataset_folders:
        scene_folders = [f for f in dataset_folder.iterdir() if f.is_dir()]
        for scene_folder in scene_folders:
            scene_labels_file = scene_folder / "gt_categories.json"
            process_scene_labels(scene_labels_file)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process scene labels and generate query mappings."
    )
    parser.add_argument(
        "--openlex_labels_folder",
        required=True,
        type=str,
        help="Path to the input JSON file.",
    )

    args = parser.parse_args()
    process_openlex_labels(args.openlex_labels_folder)
