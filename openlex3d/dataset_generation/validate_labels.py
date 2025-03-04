"""Check json structure, duplicate labels, double spaces and spell-checking. Requires the pyspellchecker package.

Spell-checking is only indicative. pyspellchecker may flag acronyms and British English spelling.
"""

from sys import argv
import json

from spellchecker import SpellChecker


def has_no_spelling_mistakes(text):
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled) == 0


def main(path):
    data = json.load(open(path, "r"))
    obj_keys = set(["name", "object_id", "labels"])
    img_attributes_keys = set(["synonyms", "vis_sim", "depictions"])
    label_keys = set(["image_attributes", "flag"])

    # Validate keys (e.g., we put flag in the wrong dict)
    # Technically doesn't cover duplicate keys in the original json
    print("Checking json structure...")
    for object in data["dataset"]["samples"]:
        if object.keys() != obj_keys:
            print(f"    {object['name']} does not have the expected keys.")

        if object["labels"].keys() != label_keys and set(object["labels"]) != set(
            ["image_attributes"]
        ):
            print(f"    {object['name']} labels does not have the expected keys.")

        if object["labels"]["image_attributes"].keys() != img_attributes_keys:
            print(
                f"    {object['name']} image_attributes does not have the expected keys."
            )

        if "flag" in object["labels"] and object["labels"]["flag"] != "ambiguous":
            print(f"    {object['name']} has wrong flag value.")

    # Make sure labels are unique
    print("Testing for unique labels...")
    for object in data["dataset"]["samples"]:
        labels = []
        for category in object["labels"]["image_attributes"].values():
            labels += category
        label_set = set(labels)
        if len(labels) != len(label_set):
            print(f"    {object['name']} does not have unique labels.")

    # Test for basic spelling errors
    print("Spell-checking...")
    for object in data["dataset"]["samples"]:
        labels = []
        for category in object["labels"]["image_attributes"].values():
            labels += category
        for label_i in labels:
            if not has_no_spelling_mistakes(label_i):
                print(
                    f"    {object['name']}: detected spelling error in label '{label_i}'"
                )
            if "  " in label_i:
                print(
                    f"    {object['name']}: detected multiple consecutive spaces in label '{label_i}'"
                )


if __name__ == "__main__":
    path = argv[1]
    main(path)
