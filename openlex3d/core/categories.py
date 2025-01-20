import json
import numpy as np

from omegaconf import DictConfig
from typing import List

__CATEGORIES = ["synonyms", "depictions", "vis_sim", "clutter", "missing", "incorrect"]


def get_categories():
    return __CATEGORIES


class CategoriesHandler:
    def __init__(self, path: str):
        with open(path, "r") as f:
            self._categories = DictConfig(json.load(f))

    def has_object(self, id: int):
        for obj in self._categories.dataset.samples[0]:
            if obj.id == id:
                return True
        else:
            return False

    def _match(self, id: int, query: str, category: str):
        if category in ["clutter", "missing", "incorrect"]:
            return 0.0
        # This checks if the query exists in the list of labels for the category
        return float(
            query
            in self._categories.dataset.samples[id].labels.image_attributes[category]
        )

    def _check_clutter(self, id: int, query: str):
        # This gets the list of indices of the clutter, and uses them to check matches
        clutter_ids = self._categories.dataset.samples[id].labels.image_attributes[
            "clutter"
        ]
        for id in clutter_ids:
            output = self._match(id, query, "synonyms")
            if output > 0.0:
                return 1.0
        return 0.0

    def match(self, id: int, query: str, category: str = "all") -> np.ndarray:
        # This is the main interface to find a match
        assert isinstance(query, str)
        assert category in get_categories() or category == "all"

        output = np.zeros(len(get_categories()))

        if category == "all":
            for i, cat in enumerate(get_categories()):
                if cat == "clutter":
                    output[i] = self._check_clutter(id, query)
                else:
                    output[i] = self._match(id, query, cat)

                if output[i] > 0.0:
                    break
        else:
            i = get_categories().index(category)
            output[i] = self._match(id, query, category)

        # If there are no matches, therefore is "incorrect"
        if np.sum(output) == 0.0:
            output[-1] = 1.0

        return output

    def batch_match(
        self, id: int, query: List[str], category: str = "all"
    ) -> np.ndarray:
        # This calls the matcher for a list of queries queries
        assert isinstance(query, list)
        M = len(query)
        output = np.zeros(M, len(get_categories()))

        for i, query in enumerate(query):
            output[i, :] = self.match(id, query, category)

        return output
