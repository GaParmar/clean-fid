import os
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
import zipfile
import cleanfid
from cleanfid.utils import *
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import *


"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""
Compute the KID score given the sets of features
"""
def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""
Compute the inception features for a batch of images
"""
def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""
def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", verbose=True):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)

    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, mode=mode)
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of image files
"""
def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True):
    # get all relevant files in the dataset
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS]
    else:
        files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description, verbose=verbose)
    return np_feats


"""
Compute the FID score given the inception features stack
"""
def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


"""
Computes the FID score for a folder of images for a specific dataset
and a specific resolution
"""
def fid_folder(fdir, dataset_name, dataset_res, dataset_split,
               model=None, mode="clean", num_workers=12,
               batch_size=128, device=torch.device("cuda"), verbose=True):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                    mode=mode, seed=0, split=dataset_split)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(fdir, model, num_workers=num_workers,
                                    batch_size=batch_size, device=device,
                                    mode=mode, description=f"FID {fbname} : ", verbose=verbose)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Compute the FID stats from a generator model
"""
def get_model_features(G, model, mode="clean", z_dim=512,
        num_gen=50_000, batch_size=128,
        device=torch.device("cuda"), desc="FID model: ", verbose=True):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    if verbose:
        pbar = tqdm(range(num_iters), desc=desc)
    else:
        pbar = range(num_iters)
    for idx in pbar:
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)))
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Computes the FID score for a generator model for a specific dataset
and a specific resolution
"""
def fid_model(G, dataset_name, dataset_res, dataset_split,
              model=None, z_dim=512, num_gen=50_000,
              mode="clean", num_workers=0, batch_size=128,
              device=torch.device("cuda"), verbose=True):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 mode=mode, seed=0, split=dataset_split,)

    # Generate test features
    np_feats = get_model_features(G, model, mode=mode,
        z_dim=z_dim, num_gen=num_gen,
        batch_size=batch_size, device=device, verbose=verbose)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""
def compare_folders(fdir1, fdir2, feat_model, mode, num_workers=0,
                    batch_size=8, device=torch.device("cuda"), verbose=True):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname1} : ", verbose=verbose)
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Test if a custom statistic exists
"""
def test_stats_exists(name, mode, metric="FID"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    if metric == "FID":
        fname = f"{name}_{mode}_{split}_{res}.npz"
    elif metric == "KID":
        fname = f"{name}_{mode}_{split}_{res}_kid.npz"
    fpath = os.path.join(stats_folder, fname)
    return os.path.exists(fpath)


"""
Remove the custom FID features from the stats folder
"""
def remove_custom_stats(name, mode="clean"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    # remove the FID stats
    split, res = "custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)
    # remove the KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}_{split}_{res}_kid.npz")
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)


"""
Cache a custom dataset statistics file
"""
def make_custom_stats(name, fdir, num=None, mode="clean",
                    num_workers=0, batch_size=64, device=torch.device("cuda")):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)

    feat_model = build_feature_extractor(mode, device)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(fdir, feat_model, num_workers=num_workers, num=num,
                                    batch_size=batch_size, device=device,
                                    mode=mode, description=f"custom stats: {fbname} : ")
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
    # KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}_{split}_{res}_kid.npz")
    print(f"saving custom KID stats to {outf}")
    np.savez_compressed(outf, feats=np_feats)


def compute_kid(fdir1=None, fdir2=None, gen=None,
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda"), dataset_name="FFHQ",
            dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    # if both dirs are specified, compute KID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute KID between two folders")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = get_folder_features(fdir1, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode,
                            description=f"KID {fbname1} : ")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(fdir2, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode,
                            description=f"KID {fbname2} : ")
        score = kernel_distance(np_feats1, np_feats2)
        return score

    # compute kid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute KID of a folder with {dataset_name} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        fbname = os.path.basename(fdir1)
        # get all inception features for folder images
        np_feats = get_folder_features(fdir1, model, num_workers=num_workers,
                                        batch_size=batch_size, device=device,
                                        mode=mode, description=f"KID {fbname} : ")
        score = kernel_distance(ref_feats, np_feats)
        return score

    # compute fid for a generator
    elif gen is not None:
        print(f"compute KID of a model with {dataset_name}-{dataset_res} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        # Generate test features
        np_feats = get_model_features(gen, model, mode=mode,
            z_dim=z_dim, num_gen=num_gen, desc="KID model: ",
            batch_size=batch_size, device=device)
        score = kernel_distance(ref_feats, np_feats)
        return score

    else:
        raise ValueError("invalid combination of directories and models entered")


def compute_fid(fdir1=None, fdir2=None, gen=None,
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda"), dataset_name="FFHQ",
            dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512,
            custom_feat_mode=None, verbose=True):
    # build the feature extractor based on the mode
    if custom_feat_mode is None:
        feat_model = build_feature_extractor(mode, device)
    else:
        feat_model = custom_feat_mode

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        if not verbose:
            print("compute FID between two folders")
        score = compare_folders(fdir1, fdir2, feat_model,
            mode=mode, batch_size=batch_size,
            num_workers=num_workers, device=device, verbose=verbose)
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        if not verbose:
            print(f"compute FID of a folder with {dataset_name} statistics")
        score = fid_folder(fdir1, dataset_name, dataset_res, dataset_split,
            model=feat_model, mode=mode, num_workers=num_workers,
            batch_size=batch_size, device=device, verbose=verbose)
        return score

    # compute fid for a generator
    elif gen is not None:
        if not verbose:
            print(f"compute FID of a model with {dataset_name}-{dataset_res} statistics")
        score = fid_model(gen, dataset_name, dataset_res, dataset_split,
                model=feat_model, z_dim=z_dim, num_gen=num_gen,
                mode=mode, num_workers=num_workers, batch_size=batch_size,
                device=device, verbose=verbose)
        return score

    else:
        raise ValueError("invalid combination of directories and models entered")
