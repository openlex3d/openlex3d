from pathlib import Path

PACKAGE_BASE_PATH = Path(__file__).absolute().parent


def get_path():
    return PACKAGE_BASE_PATH
