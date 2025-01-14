from importlib import import_module
from omegaconf import DictConfig


def load_model(config: DictConfig):
    model_type = config.type
    model_module = import_module(f"openlex3d.models.{model_type}")

    return model_module.load_model(backbone=config.type, checkpoint=config.checkpoint)
