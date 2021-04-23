import os
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
from cleanfid.utils import *
from cleanfid.features import *
from cleanfid.resize import *

"""
Compute the FID score given the mu, sigma of two sets
"""
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
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
                       use_legacy_pytorch=False,
                       use_legacy_tensorflow=False,
                       custom_fn_resize=None, 
                       description=""):
    # define the model if it is not specified
    if model is None:
        if use_legacy_pytorch:
            model = build_feature_extractor(name="pytorch_inception")
        else:
            model = build_feature_extractor(name="torchscript_inception")
    # build resizing function based on options
    if custom_fn_resize is not None:
        fn_resize = custom_fn_resize
    elif use_legacy_pytorch:
        fn_resize = make_resizer("PyTorch", False, "bilinear", (299, 299))
    elif use_legacy_tensorflow:
        fn_resize = make_resizer("TensorFlow", False, "bilinear", (299, 299))
    else:
        fn_resize = make_resizer("PIL", False, "bicubic", (299, 299))
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fn_resize=fn_resize)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    pbar = tqdm(dataloader, desc=description)
    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of features
"""
def get_folder_features(fdir, model=None, num_workers=12,
                        num=None, batch_size=128, device=torch.device("cuda"),
                        use_legacy_pytorch=False, use_legacy_tensorflow=False,
                        custom_fn_resize=None, description=""):
    # get all relevant files in the dataset
    files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"*.{ext}"))])
    if num is not None:
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device,
                                  use_legacy_pytorch=use_legacy_pytorch,
                                  use_legacy_tensorflow=use_legacy_tensorflow,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description)
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
def fid_folder(fdir, dataset_name, dataset_res,
               model=None, use_legacy_pytorch=False,
               use_legacy_tensorflow=False, num_workers=12,
               batch_size=128, device=torch.device("cuda")):
    # define the model if it is not specified
    if model is None:
        if use_legacy_pytorch:
            model = build_feature_extractor(name="pytorch_inception")
        else:
            model = build_feature_extractor(name="torchscript_inception")
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 use_legacy_tf=use_legacy_tensorflow,
                                                 use_legacy_pyt=use_legacy_pytorch, seed=0)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(fdir, model, num_workers=num_workers,
                                   batch_size=batch_size, device=device,
                                   use_legacy_pytorch=use_legacy_pytorch,
                                   use_legacy_tensorflow=use_legacy_tensorflow,
                                   description=f"FID {fbname} : ")
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid

"""
Computes the FID score for a generator model for a specific dataset 
and a specific resolution
"""
def fid_model(G, dataset_name, dataset_res,
              model=None, z_dim=512, num_fid=50_000,
              use_legacy_pytorch=False, use_legacy_tensorflow=False,
              num_workers=0, batch_size=128,
              device=torch.device("cuda")):
    # define the model if it is not specified
    if model is None:
        if use_legacy_pytorch:
            model = build_feature_extractor(name="pytorch_inception")
        else:
            model = build_feature_extractor(name="torchscript_inception")

    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 use_legacy_tf=use_legacy_tensorflow,
                                                 use_legacy_pyt=use_legacy_pytorch, seed=0)
    # build resizing function based on options
    if use_legacy_pytorch:
        fn_resize = make_resizer("PyTorch", False, "bilinear", (299, 299))
    elif use_legacy_tensorflow:
        fn_resize = make_resizer("TensorFlow", False, "bilinear", (299, 299))
    else:
        fn_resize = make_resizer("PIL", False, "bicubic", (299, 299))

    # Generate test features
    num_iters = int(np.ceil(num_fid // batch_size))
    l_feats = []

    for idx in tqdm(range(num_iters), desc=f"FID model: "):
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,1]
            img_batch = G(z_batch)

            # resize the images by wrapping the batch in a dataloader
            # ds = TensorResizeDataset(img_batch.cpu(), fn_resize)
            # dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=20, pin_memory=True)
            # resized_batch = next(iter(dl))
            resized_batch = torch.zeros(batch_size, 3, 299, 299)
            for idx in range(batch_size):
                curr_img = img_batch[idx]
                img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                #img_pil = torchvision.transforms.ToPILImage()(curr_img)
                img_resize = fn_resize(img_np)
                resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)))
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""
def compare_folders(fdir1, fdir2, num_workers=0,
                    batch_size=8, device=torch.device("cuda"),
                    use_legacy_pytorch=False,
                    use_legacy_tensorflow=False,
                    ):
    if use_legacy_pytorch:
        model = build_feature_extractor(name="pytorch_inception")
    else:
        model = build_feature_extractor(name="torchscript_inception")

    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, use_legacy_pytorch=use_legacy_pytorch,
                                    use_legacy_tensorflow=use_legacy_tensorflow, description=f"FID {fbname1} : ")
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)

    
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, use_legacy_pytorch=use_legacy_pytorch,
                                    use_legacy_tensorflow=use_legacy_tensorflow, description=f"FID {fbname2} : ")

    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid
