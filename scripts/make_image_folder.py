import os
import pdb
from tqdm import tqdm
import argparse
import torch
import torchvision
import numpy as np
from glob import glob
from cleanfid import fid

parser = argparse.ArgumentParser()
parser.add_argument("--ds", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--outf", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    if args.ds=="cifar100" and args.split=="train":
        ds = torchvision.datasets.CIFAR100(root="scripts/tmp/", train=True, download=True)
        for idx, (img,_) in tqdm(enumerate(ds)):
            fname = os.path.join(args.outf, f"{idx:06d}.png")
            img.save(fname)
    elif args.ds=="cifar100" and args.split=="test":
        ds = torchvision.datasets.CIFAR100(root="scripts/tmp/", train=False, download=True)
        for idx, (img,_) in tqdm(enumerate(ds)):
            fname = os.path.join(args.outf, f"{idx:06d}.png")
            img.save(fname)