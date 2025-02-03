import json
from collections import defaultdict
import os
import numpy as np
import re


def merge_json_files(json_files):
    image_data = defaultdict(
        lambda: {
            "name": "",
            "synonyms": set(),
            "visually_similar": set(),
            "depictions": set(),
        }
    )

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            for sample in data.get("dataset", {}).get("samples", []):
                name = sample.get("name")
                labels_exist = sample.get("labels", {}).get("ground-truth", {})

                if not labels_exist:
                    continue

                skipped = labels_exist.get("label_status", {})

                if skipped == "SKIPPED":
                    continue

                name = sample.get("name")
                labels = (
                    sample.get("labels", {})
                    .get("ground-truth", {})
                    .get("attributes", {})
                )
                image_attributes = labels.get("image_attributes", {})

                if "Synonyms" in image_attributes:
                    synonyms = image_attributes["Synonyms"].split(",")
                    image_data[name]["synonyms"].update(
                        label.strip().lower() for label in synonyms if label.strip()
                    )

                if "Visually Similar Categories" in image_attributes:
                    visually_similar = image_attributes[
                        "Visually Similar Categories"
                    ].split(",")
                    image_data[name]["visually_similar"].update(
                        label.strip().lower()
                        for label in visually_similar
                        if label.strip()
                    )

                if "Related / patterns on objects etc" in image_attributes:
                    related = image_attributes[
                        "Related / patterns on objects etc"
                    ].split(",")
                    image_data[name]["depictions"].update(
                        label.strip().lower() for label in related if label.strip()
                    )

                # Store the name for reference
                image_data[name]["name"] = name

    # Prepare the merged JSON structure
    merged_json = {
        "name": "hm3d_000890_merged",
        "dataset": {
            # "name": "replica",
            # "task_attributes": {
            #     "format_version": "0.1",
            #     "categories": [{"name": "object", "id": 1, "color": [0, 113, 188], "attributes": []}],
            #     "image_attributes": [
            #         {"name": "Synonyms", "input_type": "text", "is_mandatory": False},
            #         {"name": "Visually Similar", "input_type": "text", "is_mandatory": False},
            #         {"name": "Related / Patterns", "input_type": "text"}
            #     ]
            # },
            "samples": []
        },
    }

    # Add merged samples to the dataset
    for name, labels in image_data.items():
        match = re.search(r"\d+", name)
        object_id = int(match.group()) if match else None

        sample = {
            # "uuid": name,
            "name": labels["name"],
            "object_id": object_id,
            # "attributes": {
            #     "image": {"url": ""},
            # },
            # "metadata": {},
            "labels": {
                # "ground-truth": {
                # "label_status": "LABELED",
                # "attributes": {
                # "format_version": "0.1",
                # "annotations": [],
                # "segmentation_bitmap": {"url": ""},
                "image_attributes": {
                    "synonyms": sorted(labels["synonyms"]),
                    "vis_sim": sorted(labels["visually_similar"]),
                    "depictions": sorted(labels["depictions"]),
                }
                # }
                # }
            },
        }
        merged_json["dataset"]["samples"].append(sample)

    labels_count_per_object = {
        name: {
            "synonyms_count": len(labels["synonyms"]),
            "visually_similar_count": len(labels["visually_similar"]),
            "related_count": len(labels["depictions"]),
            "total_labels": len(labels["synonyms"])
            + len(labels["visually_similar"])
            + len(labels["depictions"]),
        }
        for name, labels in image_data.items()
    }

    total_labels_per_object = [
        counts["total_labels"] for counts in labels_count_per_object.values()
    ]
    # mean_labels = np.mean(total_labels_per_object) # commenting because it was not used
    std_labels = np.std(total_labels_per_object)
    min_labels = np.min(total_labels_per_object)
    max_labels = np.max(total_labels_per_object)

    std_labels = float(std_labels)
    min_labels = int(min_labels)
    max_labels = int(max_labels)

    # Output the statistics for analysis
    total_unique_labels = len(
        set(
            label
            for category in ["synonyms", "visually_similar", "depictions"]
            for labels in image_data.values()
            for label in labels[category]
        )
    )
    total_samples = len(merged_json["dataset"]["samples"])
    total_labels = sum(
        counts["total_labels"] for counts in labels_count_per_object.values()
    )
    average_labels_per_object = total_labels / total_samples if total_samples > 0 else 0

    print("Final Statistics:")
    print(f"  Total Unique Labels: {total_unique_labels}")
    print(f"  Total Number of Objects (Samples): {total_samples}")
    print(f"  Total Number of Labels Across All Objects: {total_labels}")
    print(f"  Average Number of Labels per Object: {average_labels_per_object:.2f}")
    print(f"  Standard Deviation of Labels per Object: {std_labels:.2f}")
    print(f"  Minimum Labels per Object: {min_labels}")
    print(f"  Maximum Labels per Object: {max_labels}")

    all_labels = sorted(
        set(
            label
            for category in ["synonyms", "visually_similar", "depictions"]
            for labels in image_data.values()
            for label in labels[category]
        )
    )

    summary = {
        "unique_labels": all_labels,
        "final_stats": {
            "total_unique_labels": total_unique_labels,
            "total_number_of_objects": total_samples,
            "average_labels_per_object": average_labels_per_object,
            "std_labels_per_object": std_labels,
            "min_labels_per_object": min_labels,
            "max_labels_per_object": max_labels,
        },
    }

    return merged_json, summary


def main():
    # Define the list of JSON files to merge
    json_files = [
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem-00890_fh-v0.1.json",
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem-00890-mb-martin-v0.1.json",
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem-00890-mb-ulli-v0.1.json",
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem-00890-maryam-v0.1.json",
    ]

    for file in json_files:
        if not os.path.isfile(file):
            print(f"Error: The file '{file}' does not exist.")
            return

    merged_data, summary = merge_json_files(json_files)

    with open(
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem_000890_merged.json", "w"
    ) as f:
        json.dump(merged_data, f, indent=4)

    with open(
        "/home/christina/openlex3d_labels/hm3d/000890/hm3dsem_000890_merged_metadata.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
