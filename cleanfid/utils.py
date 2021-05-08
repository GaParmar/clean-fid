import os
import numpy as np
import torch
import torchvision
from PIL import Image
import urllib.request
import requests
import shutil
import torch.nn.functional as F


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, size=(299, 299), fn_resize=None):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fn_resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)

        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy'}
