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
Follow the instructions in the [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) repository to convert the `tfrecord` to a folder of images. 
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