import open_clip
import torch

FEATURE_DIM = 1024


def load_model(backbone: str = "ViT-H-14", checkpoint: str = None):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=checkpoint,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    clip_model.eval()
    return clip_model, FEATURE_DIM
