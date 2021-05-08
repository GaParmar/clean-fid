# clean-fid for Evaluating Generative Models

<br>

<p align="center">
<img src="https://raw.githubusercontent.com/GaParmar/clean-fid/main/docs/images/cleanfid_demo_folders.gif" />
</p>



[**Project**](https://www.cs.cmu.edu/~clean-fid/) | [**Paper**](https://arxiv.org/abs/2104.11222) | [**Colab Demo**](https://colab.research.google.com/drive/1ElGAHvlwTilIf_3D3cw1boirCEkFsAWI?usp=sharing)


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

We provide precompute statistics for the following configurations

| Task             | Dataset   | Resolution   | split       | mode |
| :-:              | :---:     | :-:          | :-:         |  :-:  |
| Image Generation | FFHQ      | 256,1024 | `train+val` | `clean`, `legacy_pytorch`, `legacy_tensorflow`|
| Image Generation | LSUN Outdoor Churches      | 256 | `train` | `clean`, `legacy_pytorch`, `legacy_tensorflow`|
| Image to Image | horse2zebra | 128,256      | `train`, `test`, `train+test` | `clean`, `legacy_pytorch`, `legacy_tensorflow`|

**Using precomputed statistics**
In order to compute the FID score with the precomputed dataset statistics, use the corresponding options. For instance, to compute the clean-fid score on generated 256x256 FFHQ images use the command:
  ```
  fid_score = fid.compute_fid(fdir1, dataset_name="FFHQ", dataset_res=256,  mode="clean")
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

<br>

**FFHQ @ 1024x1024**
| Model     | Legacy-FID    | Clean-FID  |
| :---:     | :-:           | :-: |
| StyleGAN2 | 2.85 ± 0.05  | 3.08 ± 0.05 |
| StyleGAN  | 4.44 ± 0.04  | 4.82 ± 0.04 |
| MSG-GAN   | 6.09 ± 0.04  | 6.58 ± 0.06 |

<br>

**Image-to-Image**
(horse->zebra @ 256x256)
Computed using test images

| Model     | Legacy-FID  | Clean-FID  |
| :--:     | :-:           | :-: |
| CycleGAN  | 77.20 | 75.17 |
| CUT       | 45.51 | 43.71 |



---

## Building from source
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
