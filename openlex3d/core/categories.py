import json

from typing import List


NONE = "none"

CATEGORIES = [
    SYNONYMS := "synonyms",
    DEPICTIONS := "depictions",
    VISUALLY_SIMILAR := "vis_sim",
    CLUTTER := "clutter",
    MISSING := "missing",
    INCORRECT := "incorrect",
]

COLORS = {
    SYNONYMS: [34, 139, 34],  # green
    DEPICTIONS: [255, 255, 0],  # yellow
    VISUALLY_SIMILAR: [255, 165, 0],  # orange
    CLUTTER: [0, 0, 255],  # blue,
    MISSING: [220, 220, 220],  # blue,
    INCORRECT: [255, 0, 0],  # red
    NONE: [0, 0, 0],  # black
}


def get_categories():
    return CATEGORIES


def get_main_categories():
    return [SYNONYMS, DEPICTIONS, VISUALLY_SIMILAR]


def get_color_mapping():
    return COLORS


def get_color(category: str):
    assert category in CATEGORIES or category == NONE
    return COLORS[category]


class CategoriesHandler:
    def __init__(self, path: str, strip_spaces: bool = True):
        self.strip_spaces = strip_spaces
        with open(path, "r") as f:
            tmp_samples = (json.load(f))["dataset"]["samples"]

        self._samples = {}
        for _, sample in enumerate(tmp_samples):
            if self.strip_spaces:
                for cat in get_main_categories():
                    sample["labels"]["image_attributes"][cat] = [
                        label.replace(" ", "")
                        for label in sample["labels"]["image_attributes"][cat]
                    ]

            self._samples[sample["object_id"]] = sample

    def has_object(self, id: int):
        """Checks if an object id exists

        Args:
            id (int): id of the object to search for

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            self._get_sample(id)
            if len(self._get_labels_from_category(id, SYNONYMS)):
                return True
            else:
                # If there are no synonyms return False
                return False
        except Exception:
            # If there is no matching object return False
            return False

    def match(self, id: int, query: str) -> str:
        """Interface to match an object with a query label

        Args:
            id (int): id of the object
            query (str): predicted label for the object to match

        Returns:
            str: label of the corresponding category it matches with, 'incorrect' if not match is found
        """
        # This is the main interface to find a match
        assert isinstance(query, str)

        # Check if clutter
        if self._check_clutter(id, query):
            return CLUTTER

        # Check if any of the main categories
        for i, cat in enumerate(get_main_categories()):
            if self._match(id, query, cat):
                return cat

        # Otherwise incorrect
        return INCORRECT

    def batch_category_match(
        self, id: int, query: List[str], category: str
    ) -> List[bool]:
        """Checks if a list of labels fall into the given category

        Args:
            id (int): id of the object to match against
            query (List[str]): List of predicted labels (strings)
            category (str): Category to match agains

        Returns:
            List[bool]: List of booleans with the queries that fall into the category
        """

        # This calls the matcher for a list of queries
        matches = []
        for i, query in enumerate(query):
            matches.append(self._match(id, query, category))
        return matches

    def _get_sample(self, id: int):
        try:
            return self._samples[id]
        except Exception:
            raise IndexError(f"Object {id} not found")

    def _get_labels_from_category(self, id: int, category: str) -> List[str]:
        """This provides a simpler interface to get the labels on a category

        Args:
            id (int): id of the object to match against
            category (str): category to query

        Returns:
            List[str]: List of labels associated to the category
        """
        try:
            sample = self._get_sample(id)
            return sample["labels"]["image_attributes"][category]
        except Exception:
            return []

    def _match(self, id: int, query: str, category: str) -> bool:
        """Private method, provides a binary

        Args:
            id (int): id of the object to match against
            query (str): predicted label for the object to match
            category (str): Specific category to match against

        Returns:
            bool: _True if the query exists in the given category
        """
        if category not in get_main_categories():
            return False
        # This checks if the query exists in the list of labels for the category

        if self.strip_spaces:
            # We strip spaces to avoid negative matches due to inconsistent spacing
            query = query.replace(" ", "")

        return query in self._get_labels_from_category(id, category)

    def _check_clutter(self, id: int, query: str) -> bool:
        """Check if the query falls into the clutter category
        We consider the object as clutter

        Args:
            id (int): id of the object to match against
            query (str): predicted label for the object to match

        Returns:
            bool: True if object was classified as clutter
        """
        # This gets the list of indices of the clutter, and uses them to check matches
        clutter_ids = self._get_labels_from_category(id, CLUTTER)
        for id in clutter_ids:
            if self._match(id, query, SYNONYMS):
                return True
        return False
