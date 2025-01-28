import json
from pathlib import Path


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
                if query not in query_mapping[level]:
                    query_mapping[level][query] = []
                query_mapping[level][query].append(obj_id)

    return query_mapping


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def process_scene_labels(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    base_name = Path(input_path).stem
    scene_queries_output_path = Path(f"{base_name}_queries.json")
    query_mapping_output_path = Path(f"{base_name}_query_to_object_mapping.json")

    updated_scene_data = add_queries_to_scene(data)

    query_mapping = build_query_to_obj_mapping(updated_scene_data)

    save_json(updated_scene_data, scene_queries_output_path)
    save_json(query_mapping, query_mapping_output_path)

    print(
        f"Processed data saved to {scene_queries_output_path} and {query_mapping_output_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process scene labels and generate query mappings."
    )
    parser.add_argument(
        "--scene_labels_file",
        required=True,
        type=str,
        help="Path to the input JSON file.",
    )

    args = parser.parse_args()
    process_scene_labels(args.scene_labels_file)
