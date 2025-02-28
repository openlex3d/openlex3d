from typing import List

import clip
import numpy as np
import torch

from openlex3d.models.base import VisualLanguageEncoder


def load_model(backbone: str = "ViT-H-14", checkpoint: str = None, device: str = "cpu"):
    return OpenAIClip(backbone=backbone, checkpoint=checkpoint, device=device)

class OpenAIClip(VisualLanguageEncoder):
    FEATURE_DIM = 768

    def __init__(
        self, backbone: str = "ViT-H-14", checkpoint: str = None, device: str = "cpu"
    ):
        self._clip_model, _ = clip.load(backbone, device=device, jit=False)
        self._device = device
        self._clip_model.eval()
        self.FEATURE_DIM = self._clip_model.text_projection.shape[1]

    def compute_text_features(self, input_text: List[str], batch_size=64):
        """
        Get the text features from the CLIP model
        :param in_text (list): the text to get the features from
        :param clip_model (CLIP): the CLIP model
        :param clip_feat_dim (int): the dimension of the CLIP features
        :param batch_size (int): the batch size for the inference
        :return: the text features
        """

        text_tokens = clip.tokenize(input_text).to(self._device)
        # text_tokens = open_clip.tokenize(input_text).to(self._device)
        text_id = 0
        text_feats = np.zeros((len(input_text), self.FEATURE_DIM), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(input_text) - text_id, batch_size)
            text_batch = text_tokens[text_id : text_id + batch_size]
            with torch.inference_mode():
                batch_feats = self._clip_model.encode_text(text_batch).float()
                batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
                batch_feats = np.float32(batch_feats.cpu())
                text_feats[text_id : text_id + batch_size, :] = batch_feats
                text_id += batch_size
        return text_feats
