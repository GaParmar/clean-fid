## Flickr-Faces-HQ Dataset (FFHQ)

**@ 1024 x 1024**
First, download the dataset images at full resolution provided [here](https://github.com/NVlabs/ffhq-dataset). In particular, dowload the `images1024x1024` folder from [google drive](https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS).

Run the following command to generate statistics using all `train+val` 70k images. 
```
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images1024x1024/ \
    --output_file stats/ffhq_legacy_tensorflow_trainval70k_1024.npz \
    --mode legacy_tensorflow

python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images1024x1024/ \
    --output_file stats/ffhq_clean_trainval70k_1024.npz \
    --mode clean
```

Run the following command to generate statistics using 50k images randomly sampled from combined `train+val` splits.
```
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images1024x1024/ \
    --num_images 50000 --seed 0 --mode legacy_tensorflow \
    --output_file stats/ffhq_legacy_tensorflow_trainval_1024.npz \

python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images1024x1024/ \
    --num_images 50000 --seed 0 --mode clean \
    --output_file stats/ffhq_clean_trainval_1024.npz \
```

<br>

**@ 256 x 256** 
The FFHQ dataset provides the dataset at multiple resolution in the form on `.tfrecord` files. For 256x256 resolution, download the file `ffhq-r08.tfrecords` from [google drive](https://drive.google.com/drive/folders/1M24jfI-Ylb-k2EGhELSnxssWi9wGUokg).
Follow the instructions in the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository to convert the `tfrecord` to a folder of images. 
Run the following commands to generate statistics using different splits.
```
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images256_tfdir/ \
    --output_file stats/ffhq_legacy_tensorflow_trainval70k_256.npz \
    --mode legacy_tensorflow

python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/FFHQ/images256_tfdir/ \
    --output_file stats/ffhq_clean_trainval70k_256.npz \
    --mode clean
```

---

## LSUN - cat
Download the corresponding `lmdb` files from [here](http://dl.yf.io/lsun/objects/cat.zip). 
Next, center crop and resize 200k images from the training set using the command below. 
```
python dataset_tool.py --source=~/datasets/lsun/lmdb/cat \
    --dest= lsuncat200k.zip --transform=center-crop \
    --width=256 --height=256 --max_images=200000
```
Unzip the generated zip file into a folder of image files `lsuncat200k\`. 
Use the script below to randomly pick 50000 images and generate the corresponding Inception statistics. 
```
python scripts/process_dataset_folder.py \
    --input_folder ~/cleanfid_images/LSUN_CAT/lsuncat200k \
    --num_images 50000 --seed 0 --mode legacy_tensorflow \
    --output_file stats/lsuncat_legacy_tensorflow_train_256.npz 

python scripts/process_dataset_folder.py \
    --input_folder ~/cleanfid_images/LSUN_CAT/lsuncat200k \
    --num_images 50000 --seed 2 --mode legacy_tensorflow \
    --output_file stats/lsuncat_legacy_tensorflow_train2_256.npz 
```

---

## Cifar-10
First, download the dataset released by the original authors [here](https://www.cs.toronto.edu/~kriz/cifar.html). Convert the 50k training images to a folder of images using [dataset_tool](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py) in the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository. 
Run the following command to generate statistics on the 50k images in the `train` split and 10k images in the `test` split.
```
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/cifar10_train_images \
    --mode legacy_tensorflow \
    --output_file stats/cifar10_legacy_tensorflow_train_32.npz
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/cifar10_train_images \
    --mode clean \
    --output_file stats/cifar10_clean_train_32.npz


python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/cifar10_test_images \
    --mode legacy_tensorflow \
    --output_file stats/cifar10_legacy_tensorflow_test_32.npz
python scripts/process_dataset_folder.py \
    --input_folder ~/datasets/cifar10_test_images \
    --mode clean \
    --output_file stats/cifar10_clean_test_32.npz
```

---
