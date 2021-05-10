import os
import argparse
import torch
import numpy as np
from glob import glob
from cleanfid import fid

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--mode", required=True)
# if specified, only these many images are used
parser.add_argument("--num_images", default=-1, type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    if args.num_images==-1:
        num = None
    else:
        num = args.num_images
    np_feats = fid.get_folder_features(
        args.input_folder, num=num, shuffle=True, seed=args.seed,
        batch_size=64, device=torch.device("cuda"), mode=args.mode)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    np.savez_compressed(args.output_file, mu=mu, sigma=sigma)