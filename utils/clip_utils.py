from argparse import ArgumentParser
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import clip
import open_clip
from PIL import Image
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    # in_text = ["a {} in the scene.".format(in_text)]
    text_tokens = open_clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats