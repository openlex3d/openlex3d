import json
import os


def get_unique_labels(json_files):
    unique_labels = set()

    for file_path in json_files:
        with open(file_path, "r") as file:
            data = json.load(file)

        for sample in data.get("dataset", {}).get("samples", {}):
            image_attributes = sample.get("labels", {}).get("image_attributes", {})

            unique_labels.update(image_attributes.get("synonyms", []))
            unique_labels.update(image_attributes.get("vis_sim", []))
            unique_labels.update(image_attributes.get("depictions", []))

    return unique_labels


def main():
    # Define the list of JSON files to merge
    # json_files = [
    #     "/home/christina/openlex3d_labels/replica/office0/replica_office0_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/office1/replica_office1_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/office2/replica_office2_merged_reviewed.json",
    #     "/home/christina/openlex3d_labels/replica/office3/replica_office3_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/office4/replica_office4_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/room0/replica_room0_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/room1/replica_room1_merged_reviewed_clutter.json",
    #     "/home/christina/openlex3d_labels/replica/room2/replica_room2_merged_reviewed_clutter.json",
    # ]

    json_files = [
        "/home/christina/openlex3d_labels/replica/office4/replica_office4_merged_reviewed_clutter.json"
    ]

    for file in json_files:
        if not os.path.isfile(file):
            print(f"Error: The file '{file}' does not exist.")
            return

    unique_labels = get_unique_labels(json_files)

    print(len(unique_labels))

    output_file = "/home/christina/openlex3d_labels/replica/replica_unique_labels.txt"

    with open(output_file, "w") as file:
        for label in sorted(unique_labels):
            file.write(label + "\n")


if __name__ == "__main__":
    main()
