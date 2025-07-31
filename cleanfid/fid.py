import os
import random
import zipfile
from glob import glob
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from scipy import linalg
from torch import nn
from tqdm import tqdm

import cleanfid
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import *
from cleanfid.utils import *

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


def frechet_distance(
    mu1: ndarray, sigma1: ndarray, mu2: ndarray, sigma2: ndarray, eps: float = 1e-6
):
    """Calculate the Frechet Distance between two Gaussian distributions.

    Args:
        mu1: The mean of the first Gaussian distribution.
        mu2: The mean of the second Gaussian distribution.
        sigma1: The covariance matrix of the first Gaussian distribution.
        sigma2: The covariance matrix of the second Gaussian distribution.
        eps: Fudge factor added to the diagonal of cov estimates.

    Returns:
        The Frechet Distance between the two distributions.

    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


"""
Compute the KID score given the sets of features
"""


def kernel_distance(
    feats1: ndarray,
    feats2: ndarray,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
):
    """Calculate the Kernel Inception Distance given two sets of features."""
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


def get_batch_features(
    batch: torch.Tensor, model: nn.Module, device: Union[str, torch.device]
):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""


def get_files_features(
    l_files: List[str],
    model: Optional[torch.nn.Module] = None,
    num_workers: int = 12,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    mode: str = "clean",
    custom_fn_resize: Optional[Callable[[Any], Any]] = None,
    description: str = "",
    fdir: Optional[str] = None,
    verbose: bool = True,
    custom_image_tranform: Optional[Callable[[Any], Any]] = None,
) -> np.ndarray:
    """
    Processes a list of file paths to extract features using a specified model.

    This function wraps the images in a DataLoader for parallelizing operations such as resizing,
    and then collects features from a specified model for each batch of images.

    Args:
        l_files (List[str]): List of file paths to the images.
        model (Optional[torch.nn.Module], optional): The model to use for feature extraction. Defaults to None.
        num_workers (int, optional): Number of worker threads for DataLoader. Defaults to 12.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 128.
        device (torch.device, optional): Device to run the model on. Defaults to torch.device("cuda").
        mode (str, optional): Mode to use for processing images. Defaults to "clean".
        custom_fn_resize (Optional[Callable[[Any], Any]], optional): Custom function for resizing images. Defaults to None.
        description (str, optional): Description for the progress bar. Defaults to an empty string.
        fdir (Optional[str], optional): Directory to fetch files from. Defaults to None.
        verbose (bool, optional): Whether to show a progress bar. Defaults to True.
        custom_image_tranform (Optional[Callable[[Any], Any]], optional): Custom transformation to apply to images. Defaults to None.

    Returns:
        np.ndarray: Numpy array of concatenated features extracted from the images.
    """

    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

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


def get_folder_features(
    fdir: str,
    model: Optional[torch.nn.Module] = None,
    num_workers: int = 12,
    num: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    mode: str = "clean",
    custom_fn_resize: Optional[Callable] = None,
    description: str = "",
    verbose: bool = True,
    custom_image_tranform: Optional[Callable] = None,
) -> np.ndarray:
    """
    Extracts features from images in a specified folder or zip file using a given model.

    Args:
        fdir: Directory or zip file containing images.
        model: Model to use for feature extraction.
        num_workers: Number of worker threads to use.
        num: Number of images to process. If None, processes all images.
        shuffle: Whether to shuffle the files before processing.
        seed: Seed for random number generator, used when shuffle is True.
        batch_size: Batch size for processing.
        device: Device to use for computation (e.g., "cuda" or "cpu").
        mode: Mode of operation, affects preprocessing.
        custom_fn_resize: Custom function for resizing images.
        description: Description of the operation for logging purposes.
        verbose: Whether to print verbose messages.
        custom_image_tranform: Custom function for additional image transformations.

    Returns:
        np.ndarray: Numpy array of extracted features.
    """
    # get all relevant files in the dataset
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS]
    else:
        files = sorted(
            [
                file
                for ext in EXTENSIONS
                for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)
            ]
        )
    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(
        files,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        custom_fn_resize=custom_fn_resize,
        custom_image_tranform=custom_image_tranform,
        description=description,
        fdir=fdir,
        verbose=verbose,
    )
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


def fid_folder(
    fdir: str,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    model: Optional[torch.nn.Module] = None,
    mode: str = "clean",
    model_name: str = "inception_v3",
    num_workers: int = 12,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
    custom_image_tranform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_fn_resize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """
    Calculates the Frechet Inception Distance (FID) between images in a folder
    and a reference dataset.

    Args:
        fdir: Directory containing the images to evaluate.
        dataset_name: Name of the reference dataset to compare against.
        dataset_res: Resolution of the images in the reference dataset.
        dataset_split: The split of the reference dataset to use.
        model: The model to use for feature extraction. If None, a default model
            specified by `model_name` is used.
        mode: The mode of operation for processing images.
        model_name: Name of the model to use for feature extraction if `model` is None.
        num_workers: Number of worker threads for loading and processing images.
        batch_size: Number of images to process in each batch.
        device: The device to perform computations on.
        verbose: If True, displays a progress bar during computation.
        custom_image_tranform: A custom function for additional image transformations.
        custom_fn_resize: A custom function for resizing images.

    Returns:
        The Frechet Inception Distance (FID) score as a float.
    """
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(
        dataset_name,
        dataset_res,
        mode=mode,
        model_name=model_name,
        seed=0,
        split=dataset_split,
    )
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(
        fdir,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Compute the FID stats from a generator model
"""


def get_model_features(
    G: Callable,
    model: torch.nn.Module,
    mode: str = "clean",
    z_dim: int = 512,
    num_gen: int = 50_000,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    desc: str = "FID model: ",
    verbose: bool = True,
    return_z: bool = False,
    custom_image_tranform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_fn_resize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
    """
    Generates model features for evaluating FID (Frechet Inception Distance).

    This function generates images using a generator model, processes them
    (including resizing), and computes their features using a specified feature
    extraction model.

    Args:
        G: Callable that generates one batch of images. model: The model used
        for feature extraction. mode: Mode for resizing images. z_dim:
        Dimensionality of the latent space. num_gen: Number of images to
        generate for feature extraction. batch_size: Batch size for image
        generation and feature extraction. device: Device to run the models on.
        desc: Description for the progress bar. verbose: Whether to show a
        progress bar. return_z: Whether to return the generated latent vectors
        along with the features. custom_image_tranform: Custom function for
        additional image transformations. custom_fn_resize: Custom function for
        resizing images.

    Returns:
        Concatenated features extracted from the generated images. If `return_z`
        is True, also returns a tensor of generated latent vectors.
    """
    if custom_fn_resize is None:
        fn_resize = build_resizer(mode)
    else:
        fn_resize = custom_fn_resize

    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    latents = []
    if verbose:
        pbar = tqdm(range(num_iters), desc=desc)
    else:
        pbar = range(num_iters)
    for idx in pbar:
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            if return_z:
                latents.append(z_batch)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                l_resized_batch = []
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    if custom_image_tranform is not None:
                        img_np = custom_image_tranform(img_np)
                    img_resize = fn_resize(img_np)
                    l_resized_batch.append(
                        torch.tensor(img_resize.transpose((2, 0, 1))).unsqueeze(0)
                    )
                resized_batch = torch.cat(l_resized_batch, dim=0)
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)[:num_gen]
    if return_z:
        latents = torch.cat(latents, 0)
        return np_feats, latents
    return np_feats


"""
Computes the FID score for a generator model for a specific dataset
and a specific resolution
"""


def fid_model(
    G: Callable,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    model=None,
    model_name: str = "inception_v3",
    z_dim: int = 512,
    num_gen: int = 50_000,
    mode: str = "clean",
    num_workers: int = 0,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
    custom_image_tranform=None,
    custom_fn_resize=None,
) -> float:
    """
    Calculates the Frechet Inception Distance (FID) score for a generative model against a reference dataset.

    Args:
        G: Callable that generates one batch of images.
        dataset_name (str): The name of the dataset to use as reference.
        dataset_res (int): The resolution of the dataset images.
        dataset_split (str): The specific split of the dataset to use.
        model: The model used to extract features from images.
        model_name (str): The name of the model to use for feature extraction.
        z_dim (int): The dimension of the latent space for the generative model.
        num_gen (int): The number of images to generate for calculating FID.
        mode (str): The mode of operation, affects preprocessing.
        num_workers (int): The number of worker threads for loading data.
        batch_size (int): The batch size for generating images.
        device (torch.device): The device to run the calculations on.
        verbose (bool): If True, print progress messages.
        custom_image_tranform: Custom transformations to apply to generated images.
        custom_fn_resize: Custom function for resizing images.

    Returns:
        float: The calculated FID score.
    """

    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(
        dataset_name,
        dataset_res,
        mode=mode,
        model_name=model_name,
        seed=0,
        split=dataset_split,
    )
    # Generate features of images generated by the model
    np_feats = get_model_features(
        G,
        model,
        mode=mode,
        z_dim=z_dim,
        num_gen=num_gen,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""


def compare_folders(
    fdir1: str,
    fdir2: str,
    feat_model,
    mode: str,
    num_workers: int = 0,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
    custom_image_tranform=None,
    custom_fn_resize=None,
) -> float:
    """
    Computes the Frechet Inception Distance (FID) score between two folders of images.

    Args:
        fdir1 (str): The path to the first folder of images.
        fdir2 (str): The path to the second folder of images.
        feat_model: The model used to extract features from images.
        mode (str): The mode of operation, affects preprocessing.
        num_workers (int): The number of worker threads for loading data. Defaults to 0.
        batch_size (int): The batch size for processing images. Defaults to 8.
        device (torch.device): The device to run the calculations on. Defaults to CUDA.
        verbose (bool): If True, print progress messages. Defaults to True.
        custom_image_tranform: Custom transformations to apply to images. Defaults to None.
        custom_fn_resize: Custom function for resizing images. Defaults to None.

    Returns:
        float: The calculated FID score between the two folders.
    """
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(
        fdir1,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname1} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(
        fdir2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname2} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Test if a custom statistic exists
"""


def test_stats_exists(
    name: str, mode: str, model_name: str = "inception_v3", metric: str = "FID"
) -> bool:
    """
    Checks if the statistics file exists for a given dataset, mode, model, and metric.

    Args:
        name: The name of the dataset.
        mode: The mode of the dataset (e.g., "clean", "noisy").
        model_name: The name of the model used for feature extraction. Defaults to "inception_v3".
        metric: The metric for which the stats are checked. Can be "FID" or "KID". Defaults to "FID".

    Returns:
        True if the statistics file exists, False otherwise.
    """
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    if metric == "FID":
        fname = f"{name}_{mode}{model_modifier}_{split}_{res}.npz"
    elif metric == "KID":
        fname = f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz"
    fpath = os.path.join(stats_folder, fname)
    return os.path.exists(fpath)


"""
Remove the custom FID features from the stats folder
"""


def remove_custom_stats(
    name: str, mode: str = "clean", model_name: str = "inception_v3"
) -> None:
    """
    Removes custom FID and KID statistics files for a given dataset and mode.

    Args:
        name: The name of the dataset.
        mode: The mode of the dataset (e.g., "clean", "noisy"). Defaults to "clean".
        model_name: The name of the model used for feature extraction. Defaults to "inception_v3".

    Raises:
        Exception: If the stats file does not exist.
    """
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    # remove the FID stats
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    outf = os.path.join(
        stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower()
    )
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)
    # remove the KID stats
    outf = os.path.join(
        stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz"
    )
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)


"""
Cache a custom dataset statistics file
"""


def make_custom_stats(
    name: str,
    fdir: str,
    num: Optional[int] = None,
    mode: str = "clean",
    model_name: str = "inception_v3",
    num_workers: int = 0,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
) -> None:
    """
    Generate custom statistics (FID and KID) for a given folder of images using specified feature extraction model.

    Args:
        name (str): Name for the output statistics file.
        fdir (str): Directory containing images to compute statistics for.
        num (Optional[int]): Number of images to use. If None, use all images in the directory. Default is None.
        mode (str): Mode for feature extraction. Default is "clean".
        model_name (str): Name of the model to use for feature extraction. Default is "inception_v3".
        num_workers (int): Number of worker threads for DataLoader. Default is 0.
        batch_size (int): Batch size for processing images. Default is 64.
        device (torch.device): Device to run the computation on. Default is torch.device("cuda").
        verbose (bool): If True, print progress and debug information. Default is True.
    """
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    outf = os.path.join(
        stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower()
    )
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)
    if model_name == "inception_v3":
        feat_model = build_feature_extractor(mode, device)
        custom_fn_resize = None
        custom_image_tranform = None
    elif model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip

        clip_fx = CLIP_fx("ViT-B/32")
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
        custom_image_tranform = None
    else:
        raise ValueError(f"The entered model name - {model_name} was not recognized.")

    # get all inception features for folder images
    np_feats = get_folder_features(
        fdir,
        feat_model,
        num_workers=num_workers,
        num=num,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        mode=mode,
        description=f"custom stats: {os.path.basename(fdir)} : ",
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)

    # KID stats
    outf = os.path.join(
        stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz".lower()
    )
    print(f"saving custom KID stats to {outf}")
    np.savez_compressed(outf, feats=np_feats)


def compute_kid(
    fdir1: Optional[str] = None,
    fdir2: Optional[str] = None,
    gen: Optional[Callable] = None,
    mode: str = "clean",
    num_workers: int = 12,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    dataset_name: str = "FFHQ",
    dataset_res: int = 1024,
    dataset_split: str = "train",
    num_gen: int = 50_000,
    z_dim: int = 512,
    verbose: bool = True,
    use_dataparallel: bool = True,
) -> float:
    """
    Compute the Kernel Inception Distance (KID) between two sets of images, or between generated images and a dataset.

    Args:
        fdir1 (Optional[str]): Directory containing the first set of images. Default is None.
        fdir2 (Optional[str]): Directory containing the second set of images or reference dataset. Default is None.
        gen (Optional[Callable]): A generator function for creating images. Default is None.
        mode (str): The mode of operation for feature extraction. Default is "clean".
        num_workers (int): Number of worker threads for DataLoader. Default is 12.
        batch_size (int): Batch size for processing images. Default is 32.
        device (torch.device): The device to run the computation on. Default is torch.device("cuda").
        dataset_name (str): Name of the dataset for reference statistics. Default is "FFHQ".
        dataset_res (int): Resolution of the dataset images. Default is 1024.
        dataset_split (str): The dataset split to use. Default is "train".
        num_gen (int): Number of images to generate for computing KID. Default is 50,000.
        z_dim (int): Dimensionality of the latent space for generation. Default is 512.
        verbose (bool): If True, print progress and debug information. Default is True.
        use_dataparallel (bool): Whether to use DataParallel for feature extraction. Default is True.

    Returns:
        float: The computed KID score.

    Raises:
        ValueError: If an invalid combination of directories and models is entered.
    """
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(
        mode, device, use_dataparallel=use_dataparallel
    )

    # if both dirs are specified, compute KID between folders
    if fdir1 is not None and fdir2 is not None:
        if verbose:
            print("compute KID between two folders")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = get_folder_features(
            fdir1,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"KID {fbname1} : ",
            verbose=verbose,
        )
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(
            fdir2,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"KID {fbname2} : ",
            verbose=verbose,
        )
        score = kernel_distance(np_feats1, np_feats2)
        return score

    # compute kid of a folder
    elif fdir1 is not None and fdir2 is None:
        if verbose:
            print(f"compute KID of a folder with {dataset_name} statistics")
        ref_feats = get_reference_statistics(
            dataset_name,
            dataset_res,
            mode=mode,
            seed=0,
            split=dataset_split,
            metric="KID",
        )
        fbname = os.path.basename(fdir1)
        # get all inception features for folder images
        np_feats = get_folder_features(
            fdir1,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"KID {fbname} : ",
            verbose=verbose,
        )
        score = kernel_distance(ref_feats, np_feats)
        return score

    # compute kid for a generator, using images in fdir2
    elif gen is not None and fdir2 is not None:
        if verbose:
            print("compute KID of a model, using references in fdir2")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        ref_feats = get_folder_features(
            fdir2,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"KID {fbname2} : ",
        )
        # Generate test features
        np_feats = get_model_features(
            gen,
            feat_model,
            mode=mode,
            z_dim=z_dim,
            num_gen=num_gen,
            desc="KID model: ",
            batch_size=batch_size,
            device=device,
        )
        score = kernel_distance(ref_feats, np_feats)
        return score

    # compute fid for a generator, using reference statistics
    elif gen is not None:
        if verbose:
            print(
                f"compute KID of a model with {dataset_name}-{dataset_res} statistics"
            )
        ref_feats = get_reference_statistics(
            dataset_name,
            dataset_res,
            mode=mode,
            seed=0,
            split=dataset_split,
            metric="KID",
        )
        # Generate test features
        np_feats = get_model_features(
            gen,
            feat_model,
            mode=mode,
            z_dim=z_dim,
            num_gen=num_gen,
            desc="KID model: ",
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        score = kernel_distance(ref_feats, np_feats)
        return score

    else:
        raise ValueError("invalid combination of directories and models entered")


"""
custom_image_tranform:
    function that takes an np_array image as input [0,255] and 
    applies a custom transform such as cropping
"""


def compute_fid(
    fdir1: Optional[str] = None,
    fdir2: Optional[str] = None,
    gen: Optional[Callable] = None,
    mode: str = "clean",
    model_name: str = "inception_v3",
    num_workers: int = 12,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    dataset_name: str = "FFHQ",
    dataset_res: int = 1024,
    dataset_split: str = "train",
    num_gen: int = 50_000,
    z_dim: int = 512,
    custom_feat_extractor: Optional[Callable] = None,
    verbose: bool = True,
    custom_image_tranform: Optional[Callable] = None,
    custom_fn_resize: Optional[Callable] = None,
    use_dataparallel: bool = True,
) -> float:
    """
    Compute the Fréchet Inception Distance (FID) between two sets of images, or between generated images and a dataset.

    Args:
        fdir1 (Optional[str]): Directory containing the first set of images. Default is None.
        fdir2 (Optional[str]): Directory containing the second set of images or reference dataset. Default is None.
        gen (Optional[Callable]): A generator function for creating images. Default is None.
        mode (str): The mode of operation for feature extraction. Default is "clean".
        model_name (str): Name of the model to use for feature extraction. Default is "inception_v3".
        num_workers (int): Number of worker threads for DataLoader. Default is 12.
        batch_size (int): Batch size for processing images. Default is 32.
        device (torch.device): The device to run the computation on. Default is torch.device("cuda").
        dataset_name (str): Name of the dataset for reference statistics. Default is "FFHQ".
        dataset_res (int): Resolution of the dataset images. Default is 1024.
        dataset_split (str): The dataset split to use. Default is "train".
        num_gen (int): Number of images to generate for computing FID. Default is 50,000.
        z_dim (int): Dimensionality of the latent space for generation. Default is 512.
        custom_feat_extractor (Optional[Callable]): Custom feature extractor. Default is None.
        verbose (bool): If True, print progress and debug information. Default is True.
        custom_image_tranform (Optional[Callable]): Custom transformation function for images. Default is None.
        custom_fn_resize (Optional[Callable]): Custom function for resizing images. Default is None.
        use_dataparallel (bool): Whether to use DataParallel for feature extraction. Default is True.

    Returns:
        float: The computed FID score.

    Raises:
        ValueError: If an invalid combination of directories and models is entered.
    """
    # build the feature extractor based on the mode and the model to be used
    if custom_feat_extractor is None and model_name == "inception_v3":
        feat_model = build_feature_extractor(
            mode, device, use_dataparallel=use_dataparallel
        )
    elif custom_feat_extractor is None and model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip

        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        if verbose:
            print("compute FID between two folders")
        score = compare_folders(
            fdir1,
            fdir2,
            feat_model,
            mode=mode,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            custom_image_tranform=custom_image_tranform,
            custom_fn_resize=custom_fn_resize,
            verbose=verbose,
        )
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        if verbose:
            print(f"compute FID of a folder with {dataset_name} statistics")
        score = fid_folder(
            fdir1,
            dataset_name,
            dataset_res,
            dataset_split,
            model=feat_model,
            mode=mode,
            model_name=model_name,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        return score

    # compute fid for a generator, using images in fdir2
    elif gen is not None and fdir2 is not None:
        if verbose:
            print("compute FID of a model, using references in fdir2")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(
            fdir2,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"FID {fbname2} : ",
            verbose=verbose,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
        )
        mu2 = np.mean(np_feats2, axis=0)
        sigma2 = np.cov(np_feats2, rowvar=False)
        # Generate test features
        np_feats = get_model_features(
            gen,
            feat_model,
            mode=mode,
            z_dim=z_dim,
            num_gen=num_gen,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )

        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        fid = frechet_distance(mu, sigma, mu2, sigma2)
        return fid

    # compute fid for a generator, using reference statistics
    elif gen is not None:
        if verbose:
            print(
                f"compute FID of a model with {dataset_name}-{dataset_res} statistics"
            )
        score = fid_model(
            gen,
            dataset_name,
            dataset_res,
            dataset_split,
            model=feat_model,
            model_name=model_name,
            z_dim=z_dim,
            num_gen=num_gen,
            mode=mode,
            num_workers=num_workers,
            batch_size=batch_size,
            custom_image_tranform=custom_image_tranform,
            custom_fn_resize=custom_fn_resize,
            device=device,
            verbose=verbose,
        )
        return score

    else:
        raise ValueError("invalid combination of directories and models entered")
