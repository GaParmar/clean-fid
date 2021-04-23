"""
helpers for extractign features from image
"""
import os
import numpy as np
import torch
import torch.nn as nn
from cleanfid.downloads_helper import *
from cleanfid.inception_pytorch import InceptionV3


class InceptionV3W(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

    path: locally saved inception weights
    """

    def __init__(self, path, download=True):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        if download:
            check_download_inception(fpath=path)
        path = os.path.join(path, "inception-2015-12-05.pt")
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """

    def forward(self, x):
        bs = x.shape[0]
        # make sure it is resized already
        assert x.shape[2] == 299
        # apply normalization
        x1 = x - 128
        x2 = x1 / 128
        features = self.layers.forward(x2, ).view((bs, 2048))
        return features


def build_feature_extractor(name="torchscript_inception"):
    """
    returns a functions that takes an image in range [0,1]
    and outputs a feature embedding vector
    """
    if name == "torchscript_inception":
        model = InceptionV3W("/tmp", download=True).cuda()
        def model_fn(x): return model(x * 255.0)
    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).cuda()
        model.eval()
        def model_fn(x): return model(x)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


def get_reference_statistics(name, res, use_legacy_tf=False, use_legacy_pyt=False, seed=0):
    base_url = "http://www.andrew.cmu.edu/user/gparmar/CleanFID/stats"
    if name == "FFHQ":
        if use_legacy_tf:
            rel_url = f"FFHQ_{res}_Legacy_TF_FID_{seed}.npz"
        elif use_legacy_pyt:
            rel_url = f"FFHQ_{res}_Legacy_PyT_FID_{seed}.npz"
        else:
            rel_url = f"FFHQ_{res}_CleanFID_{seed}.npz"
    else:
        raise ValueError(f"{name}_{res} statistics are not computed yet")
    url = f"{base_url}/{rel_url}"
    fpath = check_download_url(local_folder="/tmp", url=url)
    stats = np.load(fpath)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma
