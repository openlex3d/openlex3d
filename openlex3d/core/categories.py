import json


class CategoriesManager:
    def __init__(self, path: str):
        with open(path, "r") as f:
            self._categories = json.load(f)
    