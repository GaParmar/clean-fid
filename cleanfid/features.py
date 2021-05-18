"""
helpers for extractign features from image
"""
import os
import numpy as np
import torch
import torch.nn as nn
import cleanfid
from cleanfid.downloads_helper import *
from cleanfid.inception_pytorch import InceptionV3


class InceptionV3W(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

    path: locally saved inception weights
    """

    def __init__(self, path, download=True, resize_inside=False):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        if download:
            check_download_inception(fpath=path)
        path = os.path.join(path, "inception-2015-12-05.pt")
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers
        self.resize_inside=resize_inside

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """

    def forward(self, x):
        bs = x.shape[0]
        if self.resize_inside:
            features = self.base(x, return_features=True).view((bs, 2048))
        else:
            # make sure it is resized already
            assert x.shape[2] == 299
            # apply normalization
            x1 = x - 128
            x2 = x1 / 128
            features = self.layers.forward(x2, ).view((bs, 2048))
        return features


"""
returns a functions that takes an image in range [0,255]
and outputs a feature embedding vector
"""
def feature_extractor(name="torchscript_inception", device=torch.device("cuda"), resize_inside=False):
    if name == "torchscript_inception":
        model = InceptionV3W("/tmp", download=True, resize_inside=resize_inside).to(device)
        model.eval()
        def model_fn(x): return model(x)
    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()
        def model_fn(x): return model(x/255)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


def build_feature_extractor(mode, device=torch.device("cuda")):
    if mode=="legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device)
    elif mode=="legacy_tensorflow":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=True, device=device)
    elif mode=="clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device)
    return feat_model

def get_reference_statistics(name, res, mode="clean", seed=0, split="test", metric="FID"):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    if split=="custom": res = "na"
    if metric=="FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric=="KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        return stats["feats"]
