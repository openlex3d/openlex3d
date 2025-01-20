___CATEGORIES_COLORS = {
    "none": [220, 220, 220],  # grey
    "synonyms": [34, 139, 34],  # green
    "vis_sim": [255, 255, 0],  # yellow
    "related": [255, 165, 0],  # orange
    "incorrect": [255, 0, 0],  # red
    "missing": [0, 0, 0],  # black
}


def get_categories_color_mapping():
    return ___CATEGORIES_COLORS


def get_category_color(category: str):
    try:
        return ___CATEGORIES_COLORS[category]
    except Exception:
        raise ValueError(f"Category [{category}] does not have valid color")
