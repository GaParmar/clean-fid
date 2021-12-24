"""
helpers for extracting features from image
"""
import os
import numpy as np
import torch
import cleanfid
from cleanfid.downloads_helper import check_download_url
from cleanfid.inception_pytorch import InceptionV3
from cleanfid.inception_torchscript import InceptionV3W


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


"""
Build a feature extractor for each of the modes
"""
def build_feature_extractor(mode, device=torch.device("cuda")):
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device)
    elif mode == "legacy_tensorflow":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=True, device=device)
    elif mode == "clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device)
    return feat_model


"""
Load precomputed reference statistics for commonly used datasets
"""
def get_reference_statistics(name, res, mode="clean", seed=0, split="test", metric="FID"):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    if split == "custom":
        res = "na"
    if metric == "FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric == "KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        return stats["feats"]
