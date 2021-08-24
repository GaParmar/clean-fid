import os, sys, pdb
import torch, torchvision
import numpy as np
from PIL import Image
from cleanfid.fid import *
from cleanfid.inception_pytorch import InceptionV3



def get_all_acts(img_path):
    model = InceptionV3(output_blocks=[3], resize_input=False).cuda()
    acts = []
    x = (torchvision.transforms.ToTensor()(Image.open(img_path)).unsqueeze(0).cuda()-0.5)/0.5
    for idx, block in enumerate(model.blocks[:]):
        #x = block(x)
        #acts.append(x.detach().clone())
        # expand the block
        for pidx, sub_block in enumerate(block):
            x = sub_block(x)
            acts.append([idx, pidx, x.detach().clone()])
    return acts

def cmp_all_acts(acts1, acts2):
    nc = acts1.shape[1]
    l1=[]
    for i in range(0,nc - nc%3,3): l1.append(acts1[0,i:i+3,:].detach())
    l2 = [acts2[0,i:i+3,:].detach() for i in range(0,nc-nc%3,3)]
    l3 = [(l2[i]-l1[i])**2 for i in range(len(l1))]
    #pdb.set_trace()
    torchvision.utils.save_image(l1+l2+l3, "acts.png", nrow=len(l1))

if __name__=="__main__":
    path = "/home/gparmar/datasets/FFHQ/images1024x1024/27222.png"

    img_1024_png = Image.open(path)
    Image.open(path).save("_.jpg", quality=25)
    img_1024_jpg = Image.open("_.jpg")

    left,top,right,bottom = 500,500,550,550
    img_1024_png.crop((left, top, right, bottom)).save("_cropped_png.png")
    img_1024_jpg.crop((left, top, right, bottom)).save("_cropped_jpg.png")
    l_acts_png = get_all_acts("_.png")
    l_acts_jpg = get_all_acts("_.jpg")

    cmp_all_acts(l_acts_png[0][-1], l_acts_jpg[0][-1])
