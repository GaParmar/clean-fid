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


def build_feature_extractor(name="torchscript_inception", device=torch.device("cuda")):
    """
    returns a functions that takes an image in range [0,1]
    and outputs a feature embedding vector
    """
    if name == "torchscript_inception":
        model = InceptionV3W("/tmp", download=True).to(device)
        def model_fn(x): return model(x * 255.0)
    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()
        def model_fn(x): return model(x)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


def get_reference_statistics(name, res, mode="clean", seed=0, split="test"):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    if name == "FFHQ":
        if mode=="legacy_tensorflow":
            rel_url = f"FFHQ_{res}_Legacy_TF_FID_{seed}.npz"
        elif mode=="legacy_pytorch":
            rel_url = f"FFHQ_{res}_Legacy_PyT_FID_{seed}.npz"
        elif mode=="clean":
            rel_url = f"FFHQ_{res}_CleanFID_{seed}.npz"
        else:
            raise ValueError(f"{mode} mode is not implemented")
    elif name == "lsun_church":
        if mode=="legacy_tensorflow":
            rel_url = f"{name}_LegacyTF_fid_{split}_{seed}_{res}.npz"
        elif mode=="legacy_pytorch":
            rel_url = f"{name}_LegacyPyt_fid_{split}_{seed}_{res}.npz"
        elif mode=="clean":
            rel_url = f"{name}_cleanfid_{split}_{seed}_{res}.npz"
        else:
            raise ValueError(f"{mode} mode is not implemented")
    elif name == "horse2zebra" and res==256:
        if mode=="legacy_tensorflow":
            rel_url = f"horse2zebra_LegacyTF_fid_{split}.npz"
        elif mode=="legacy_pytorch":
            rel_url = f"horse2zebra_LegacyPyt_fid_{split}.npz"
        elif mode=="clean":
            rel_url = f"horse2zebra_cleanfid_{split}.npz"
        else:
            raise ValueError(f"{mode} mode is not implemented")
    elif name == "horse2zebra" and res==128:
        if mode=="legacy_tensorflow":
            rel_url = f"horse2zebra_LegacyTF_fid_{split}_128.npz"
        elif mode=="legacy_pytorch":
            rel_url = f"horse2zebra_LegacyPyt_fid_{split}_128.npz"
        elif mode=="clean":
            rel_url = f"horse2zebra_cleanfid_{split}_128.npz"
        else:
            raise ValueError(f"{mode} mode is not implemented")
    else:
        raise ValueError(f"{name}_{res} statistics are not computed yet")
    url = f"{base_url}/{rel_url}"
    mod_path = os.path.dirname(cleanfid.__file__)
    stats_folder = os.path.join(mod_path, "stats")
    fpath = check_download_url(local_folder=stats_folder, url=url)
    stats = np.load(fpath)
    mu, sigma = stats["mu"], stats["sigma"]
    return mu, sigma
