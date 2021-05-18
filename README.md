# clean-fid for Evaluating Generative Models

<br>

<p align="center">
<img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/cleanfid_demo_folders.gif" />
</p>



[**Project**](https://www.cs.cmu.edu/~clean-fid/) | [**Paper**](https://arxiv.org/abs/2104.11222) | [**Colab Demo**](https://colab.research.google.com/drive/1ElGAHvlwTilIf_3D3cw1boirCEkFsAWI?usp=sharing) | [**Leaderboard**](#cleanfid-leaderboard-for-common-tasks)


The FID calculation involves many steps that can produce inconsistencies in the final metric. As shown below, different implementations use different low-level image quantization and resizing functions, the latter of which are often implemented incorrectly.

<p align="center">
  <img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/resize_circle.png"  width="800" />
</p>


We provide an easy-to-use library to address the above issues and make the FID scores comparable across different methods, papers, and groups.

![FID Steps](https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/fid_steps.jpg)


---

[On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation](https://www.cs.cmu.edu/~clean-fid/) <br>
 [Gaurav Parmar](https://gauravparmar.com/), [Richard Zhang](https://richzhang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
arXiv 2104.11222, 2021 <br>
CMU and Adobe<br>

---


<br>

**Buggy Resizing Operations** <br>



  The definitions of resizing functions are mathematical and <em>should never be a function of the library being used</em>. Unfortunately, implementations differ across commonly-used libraries.  They are often implemented incorrectly by popular libraries. Try out the different resizing implementations in the Google colab notebook [here](https://colab.research.google.com/drive/1Q-N94S2mnLsFLpuT7WwY6d5WxGVWLGpg?usp=sharing).

  <img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/resize_circle_extended.png"  width="800" />
<br>

The inconsistencies among implementations can have a drastic effect of the evaluations metrics. The table below shows that FFHQ dataset images resized with  bicubic implementation from other libraries (OpenCV, PyTorch, TensorFlow, OpenCV) have a large FID score (≥ 6) when compared to the same images resized with the correctly implemented PIL-bicubic filter. Other correctly implemented filters from PIL (Lanczos, bilinear, box) all result in relatively smaller FID score (≤ 0.75). Note that since TF 2.0, the new flag `antialias` (default: `False`) can produce results close to PIL. However, it was not used in the existing TF-FID repo and set as `False` by default.

 <p align="center"><img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/table_resize_sc.png"  width="500" /></p>

**JPEG Image Compression**

  Image compression can have a surprisingly large effect on FID.  Images are perceptually indistinguishable from each other but have a large FID score. The FID scores under the images are calculated between all FFHQ images saved using the corresponding JPEG format and the PNG format.

<p align="center">
  <img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/jpeg_effects.png"  width="800" />
</p>

Below, we study the effect of JPEG compression for StyleGAN2 models trained on the FFHQ dataset (left) and LSUN outdoor Church dataset (right). Note that LSUN dataset images were collected with JPEG compression (quality 75), whereas FFHQ images were collected as PNG. Interestingly, for LSUN dataset, the best FID score (3.48) is obtained when the generated images are compressed with JPEG quality 87.

<p align="center">
  <img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/jpeg_plots.png"  width="800" />
</p>

---

## Quick Start


- install requirements
    ```
    pip install -r requirements.txt
    ```
- install the library
    ```
    pip install clean-fid
    ```
- Compute FID between two image folders
    ```
    from cleanfid import fid

    score = fid.compute_fid(fdir1, fdir2)
    ```


- Compute FID between one folder of images and pre-computed datasets statistics (e.g., `FFHQ`)
    ```
    from cleanfid import fid

    score = fid.compute_fid(fdir1, dataset_name="FFHQ", dataset_res=1024)

    ```

- Compute FID using a generative model and pre-computed dataset statistics:
    ```
    from cleanfid import fid

    # function that accepts a latent and returns an image in range[0,255]
    gen = lambda z: GAN(latent=z, ... , <other_flags>)

    score = fid.compute_fid(gen=gen, dataset_name="FFHQ",
            dataset_res=256, num_gen=50_000)

    ```

---
### Supported Precomputed Datasets

We provide precompute statistics for the following commonly used configurations

| Task             | Dataset   | Resolution | Reference Split          | # Reference Images | mode |
| :-:              | :---:     | :-:        | :-:            |  :-:          | :-: |
| Image Generation | `cifar10`     | 32         | `train`        |  50,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `cifar10`     | 32         | `test`         |  10,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `ffhq`        | 1024, 256  | `trainval`     |  50,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `ffhq`        | 1024, 256  | `trainval70k`  |  70,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `lsun_church` | 256        | `train`        |  50,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `lsun_horse`  | 256        | `train`        |  50,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `lsun_cat`    | 256        | `train`        |  50,000       |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Image Generation | `lsun_cat`    | 256        | `trainfull`    |  1,657,264    |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Few Shot Generation | `afhq_cat`  | 512        | `train`       |  5153         |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Few Shot Generation | `afhq_dog`  | 512        | `train`       |  4739         |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Few Shot Generation | `afhq_wild` | 512        | `train`       |  4738         |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Few Shot Generation | `brecahad`  | 512        | `train`       |  1944         |`clean`, `legacy_tensorflow`, `legacy_pytorch`|
| Few Shot Generation | `metfaces`  | 1024       | `train`       |  1336         |`clean`, `legacy_tensorflow`, `legacy_pytorch`|

<!-- | Image to Image | horse2zebra | 256      | `train`, `test`, `train+test` | `clean`, `legacy_pytorch`, `legacy_tensorflow`|
| Image to Image | cat2dog     | 256      | `train`, `test`, `train+test` | `clean`, `legacy_pytorch`, `legacy_tensorflow`| -->


**Using precomputed statistics**
In order to compute the FID score with the precomputed dataset statistics, use the corresponding options. For instance, to compute the clean-fid score on generated 256x256 FFHQ images use the command:
  ```
  fid_score = fid.compute_fid(fdir1, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
  ```

---
### Create Custom Dataset Statistics
- *dataset_path*: folder where the dataset images are stored
- *custom_name*: name to be used for the statistics
- Generating custom statistics (saved to local cache)
  ```
  from cleanfid import fid
  fid.make_custom_stats(custom_name, dataset_path, mode="clean")
  ```

- Using the generated custom statistics
  ```
  from cleanfid import fid
  score = fid.compute_fid("folder_fake", dataset_name=custom_name,
            mode="clean", dataset_split="custom")
  ```

- Removing the custom stats
  ```
  from cleanfid import fid
  fid.remove_custom_stats(custom_name, mode="clean")
  ```


---
## Backwards Compatibility

We provide two flags to reproduce the legacy FID score.

- `mode="legacy_pytorch"` <br>
    This flag is equivalent to using the popular PyTorch FID implementation provided [here](https://github.com/mseitzer/pytorch-fid/)
    <br>
    The difference between using clean-fid with this option and [code](https://github.com/mseitzer/pytorch-fid/) is **~2e-06**
    <br>
    See [doc](https://github.com/GaParmar/clean-fid/blob/main/docs/pytorch_fid.md) for how the methods are compared


- `mode="legacy_tensorflow"` <br>
    This flag is equivalent to using the official [implementation of FID](https://github.com/bioinf-jku/TTUR) released by the authors.
    <br>
    The difference between using clean-fid with this option and [code](https://github.com/bioinf-jku/TTUR) is **~2e-05**
    <br>
  See [doc](https://github.com/GaParmar/clean-fid/blob/main/docs/tensorflow_fid.md) for detailed steps for how the methods are compared

---

## CleanFID Leaderboard for common tasks

We compute the FID scores using the corresponding methods used in the original papers and using the Clean-FID proposed here. 
All values are computed using 10 evaluation runs. 

**CIFAR-10**
| Model	| Checkpoint	| Reported-FID	| Legacy-FID (reproduced)	| Clean-FID	| Reference Split	| # reference images used 	| # generated images used	| dataset_name	| dataset_res	| task_name	|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| stylegan2-mirror-flips (100%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10.pkl)	| 11.07	| 11.07 ± 0.10	| 12.96 ± 0.07	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|
| stylegan2-diff-augment (100%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10.pkl)	| 9.89	| 9.90 ± 0.09	| 10.85 ± 0.10	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|
| stylegan2-mirror-flips (20%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10-0.2.pkl)	| 23.08	| 23.01 ± 0.19	| 29.49 ± 0.17	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|
| stylegan2-diff-augment (20%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10-0.2.pkl)	| 12.15	| 12.12 ± 0.15	| 14.18 ± 0.13	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|
| stylegan2-mirror-flips (10%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10-0.1.pkl)	| 36.02	| 35.94 ± 0.17	| 43.60 ± 0.17	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|
| stylegan2-diff-augment (10%)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10-0.1.pkl)	| 14.50	| 14.53 ± 0.12	| 16.98 ± 0.18	| test	| 10000	| 10000	| cifar10	| 32	| few_shot_generation	|


<br>

**FFHQ @ 1024x1024**

| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
 | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | 2.84 | 2.86 ± 0.025 | 3.07 ± 0.025 | trainval | 50,000 | 50,000 
 | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | N/A | 2.76 ± 0.025 | 2.98 ± 0.025 | trainval70k | 50,000 | 70,000 
 
 **LSUN Categories**

| Category | Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used | # reference images used | 
|:-: | :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
Outdoor Churches | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl) | 3.86 | 3.87 ± 0.029 | 4.08 ± 0.028 | train | 50,000 | 50,000 
Horses | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl) | 3.43 | 3.41 ± 0.021 | 3.62 ± 0.023 | train | 50,000 | 50,000 
Cat | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl) | 6.93 | 7.02 ± 0.039 | 7.47 ± 0.035 | train | 50,000 | 50,000 

**FFHQ @ 256x256 (Few Show Generation)**
| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| stylegan2 1k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-1k.pkl) | 62.16 | 62.14 ± 0.108 | 64.17 ± 0.113 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 1k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-1k.pkl) | 25.66 | 25.60 ± 0.071 | 27.26 ± 0.077 | trainval70k | 50,000 | 70,000 |
| stylegan2 5k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-5k.pkl) | 26.60 | 26.64 ± 0.086 | 28.17 ± 0.090 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 5k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-5k.pkl) | 10.45 | 10.45 ± 0.047 | 10.99 ± 0.050 | trainval70k | 50,000 | 70,000 |
| stylegan2 10k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-10k.pkl) | 14.75 | 14.88 ± 0.070 | 16.04 ± 0.078 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 10k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-10k.pkl) | 7.86 | 7.82 ± 0.045 | 8.12 ± 0.044 | trainval70k | 50,000 | 70,000 |
| stylegan2 30k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-30k.pkl) | 6.16 | 6.14 ± 0.064 | 6.49 ± 0.068 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 30k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-30k.pkl) | 5.05 | 5.07 ± 0.030 | 5.18 ± 0.032 | trainval70k | 50,000 | 70,000 |

<br>

**LSUN CAT (Few Shot Generation)**
| Model	| Checkpoint | Reported-FID	| Legacy-FID (reproduced)	| Clean-FID	| Reference Split	| # reference images used	| # generated images used	| dataset_name	| dataset_res	| task_name	|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| stylegan2-mirror-flips (30k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-lsun-cat-30k.pkl)	| 10.12	| 10.15 ± 0.04	| 10.87 ± 0.04	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-diff-augment (30k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-lsun-cat-30k.pkl)	| 9.68	| 9.70 ± 0.07	| 10.25 ± 0.07	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-mirror-flips (10k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-lsun-cat-10k.pkl)	| 17.93	| 17.98 ± 0.09	| 18.71 ± 0.09	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-diff-augment (10k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-lsun-cat-10k.pkl)	| 12.07	| 12.04 ± 0.08	| 12.53 ± 0.08	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-mirror-flips (5k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-lsun-cat-5k.pkl)	| 34.69	| 34.66 ± 0.12	| 35.85 ± 0.12	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-diff-augment (5k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-lsun-cat-5k.pkl)	| 16.11	| 16.11 ± 0.09	| 16.79 ± 0.09	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-mirror-flips (1k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-lsun-cat-1k.pkl)	| 182.85	| 182.80 ± 0.21	| 185.86 ± 0.21	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|
| stylegan2-diff-augment (1k)	| [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-lsun-cat-1k.pkl)	| 42.26	| 42.07 ± 0.16	| 43.12 ± 0.16	| trainfull	| 1657264	| 50000	| lsun_cat	| 256	| few_shot_generation	|

<br>

**AFHQ**

| Model	| Checkpoint	| Reported-FID	| Legacy-FID (reproduced)	| Clean-FID	| Reported-KID (x 10^3)	| Legacy-KID (reproduced) (x 10^3)	| Clean-KID (x 10^3)	| Reference Split	| # reference images used	| # generated images used	| dataset_name	| dataset_res	| task_name	|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| stylegan2	| [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11a-small-datasets/afhqdog-mirror-stylegan2-noaug.pkl)	| 19.37	| 19.34 ± 0.08	| 20.10 ± 0.08	| 9.62	| 9.56 ± 0.12	| 10.21 ± 0.11	| train	| 4739	| 50000	| afhq_dog	| 512	| few_shot_generation	|


<br>

**Horse2Zebra (Image to Image)**
| Model     | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| CUT     | 45.5 | 45.51 | 43.71 | test | 120 | 140 |
| FastCUT | 73.4 | 73.38 | 72.53 | test | 120 | 140 |

<br>

**Cat2Dog (Image to Image)**
| Model     | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| CUT     | 76.2 | 76.21 | 77.58 | test | 500 | 500 |
| FastCUT | 94.0 | 93.95 | 95.37 | test | 500 | 500 |


---

## Building clean-fid locally from source
   ```
   python setup.py bdist_wheel
   pip install dist/*
   ```

---

## Citation

If you find this repository useful for your research, please cite the following work.
```
@article{parmar2021cleanfid,
  title={On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation},
  author={Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2104.11222},
  year={2021}
}
```

---

### Related Projects
[torch-fidelity](https://github.com/toshas/torch-fidelity): High-fidelity performance metrics for generative models in PyTorch. <br>
[TTUR](https://github.com/bioinf-jku/TTUR): Two time-scale update rule for training GANs. <br>
[LPIPS](https://github.com/richzhang/PerceptualSimilarity): Perceptual Similarity Metric and Dataset. <br>


### Credits
[PyTorch-StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) ([LICENSE](https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE))

[PyTorch-FID](https://github.com/mseitzer/pytorch-fid/) ([LICENSE](https://github.com/mseitzer/pytorch-fid/blob/master/LICENSE))

[StyleGAN2](https://github.com/NVlabs/stylegan2) ([LICENSE](https://nvlabs.github.io/stylegan2/license.html))

converted FFHQ weights: [code](https://github.com/eladrich/pixel2style2pixel) |  [LICENSE](https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE)
