import os, sys, pdb
import torch, torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from cleanfid.fid import *
from cleanfid.resize import *
from cleanfid.inception_pytorch import InceptionV3
from main_jpeg import *

path = "/home/gparmar/datasets/FFHQ/images1024x1024/27222.png"
l,t,r,b     = (40, 80, 100, 140)
rl,rt,rr,rb = (10, 20, 25, 35)

R = make_resizer("PyTorch", True, "bicubic", 256)
img_256_pil = Image.open(path).resize((256,256), resample=Image.BICUBIC)
img_256_pyt = Image.fromarray(R(np.array(Image.open(path))))

img_256_pil.save("_bicubic_pil.png")
img_256_pyt.save("_bicubic_pytorch.png")
img_256_pil.crop((l,t,r,b)).save("_cropped_bicubic_pil.png")
img_256_pyt.crop((l,t,r,b)).save("_cropped_bicubic_pytorch.png")

l_acts_a = get_all_acts("_bicubic_pil.png")
l_acts_b = get_all_acts("_bicubic_pytorch.png")

# find channels that are changing the most
l_mse = []
# for i in range(len(l_acts_a)):
for i in range(4):
    acts_block_a, acts_block_b = l_acts_a[i][-1], l_acts_b[i][-1]
    for j in range(acts_block_a.shape[1]):
        #diff = (acts_block_a[:,j,rt:rb, rl:rr] - acts_block_b[:,j,rt:rb, rl:rr]).mean()

        diff = F.mse_loss(acts_block_a[:,j,rt:rb, rl:rr], acts_block_b[:,j,rt:rb, rl:rr]).item()
        l_mse.append((diff, i, j))
l_mse.sort(key=lambda tup: tup[0]*-1) # descending order

# loop at top N channels
for i in range(10):
    diff, _i, _j = l_mse[i]
    a_act = l_acts_a[_i][-1][:,_j,:,:].cpu().numpy()
    b_act = l_acts_b[_i][-1][:,_j,:,:].cpu().numpy()
    a_act_crop = l_acts_a[_i][-1][:,_j,:,:].cpu().numpy()[0][rt:rb, rl:rr]
    b_act_crop = l_acts_b[_i][-1][:,_j,:,:].cpu().numpy()[0][rt:rb, rl:rr]
    #pdb.set_trace()
    nh,nw = a_act_crop.shape
    maxval = max(a_act.max(), b_act.max())
    n="Blues"
    plt.axis("off"), plt.imshow(a_act_crop.reshape(nh,nw), vmin=0, vmax=maxval, cmap=n)
    plt.savefig(f"acts/top-{i}_a_cropped.png", bbox_inches="tight", pad_inches = 0), plt.close()
    plt.axis("off"), plt.imshow(b_act_crop.reshape(nh,nw), vmin=0, vmax=maxval, cmap=n)
    plt.savefig(f"acts/top-{i}_b_cropped.png", bbox_inches="tight", pad_inches = 0), plt.close()
    _,nh,nw = a_act.shape
    plt.axis("off"), plt.imshow(a_act.reshape(nh,nw), vmin=0, vmax=maxval, cmap=n)
    plt.colorbar(), plt.savefig(f"acts/top-{i}_a.png", bbox_inches="tight", pad_inches = 0), plt.close()
    plt.axis("off"), plt.imshow(b_act.reshape(nh,nw), vmin=0, vmax=maxval, cmap=n)
    plt.colorbar(), plt.savefig(f"acts/top-{i}_b.png", bbox_inches="tight", pad_inches = 0), plt.close()


    
    #plt.imsave(f"acts/top-{i}_b.png", b_act_crop.reshape(nh,nw), vmin=0, vmax=maxval)
    #torchvision.utils.save_image([a_act_crop], f"acts/top-{i}_a.png")
    #torchvision.utils.save_image([b_act_crop], f"acts/top-{i}_b.png")

    # torchvision.utils.save_image([a_act, b_act, (a_act-b_act)**2], f"acts/resize_{i}.png")