
## Image Generation

**FFHQ @ 1024x1024**

| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
 | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | 2.84 | 2.86 +- 0.025 | 3.07 +- 0.025 | trainval | 50,000 | 70,000 
 | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | N/A | 2.76 +- 0.025 | 2.98 +- 0.025 | trainval70k | 50,000 | 70,000 

<!-- | StyleGAN2 | 2.85 ± 0.05  | 3.08 ± 0.05 |
| StyleGAN  | 4.44 ± 0.04  | 4.82 ± 0.04 |
| MSG-GAN   | 6.09 ± 0.04  | 6.58 ± 0.06 | -->

<!-- **FFHQ @ 256x256** -->

**LSUN Categories**

| Category | Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used | 
|:-: | :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
Outdoor Churches | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl) | 3.86 | 3.87 +- 0.029 | 4.08 +- 0.028 | train | 50,000 | 50,000 
Horses | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl) | 3.43 | 3.41 +- 0.021 | 3.62 +- 0.023 | train | 50,000 | 50,000 
Cat | stylegan2 | [ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl) | 6.93 | 7.02 +- 0.039 | 7.47 +- 0.035 | train | 50,000 | 50,000 

**CIFAR 10**
| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| stylegan2 (w/ mirror-flips) | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10.pkl) | 11.07 | 11.08 +- 0.080 | 12.94 +- 0.060 | test | 10,000 | 10,000 |
| stylegan2 (w/ diff-augment) | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10.pkl) | 9.89 | 9.90 +- 0.100 | 10.87 +- 0.103 | test | 10,000 | 10,000 |


## Image to Image

**Horse2Zebra**
| Model     | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| CUT     | 45.5 | 45.51 | 43.71 | test | 120 | 140 |
| FastCUT | 73.4 | 73.38 | 72.53 | test | 120 | 140 |

**Cat2Dog**
| Model     | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| CUT     | 76.2 | 76.21 | 77.58 | test | 500 | 500 |
| FastCUT | 94.0 | 93.95 | 95.37 | test | 500 | 500 |

## Few Shot Image Generation

**FFHQ @ 256x256**
| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| stylegan2 1k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-1k.pkl) | 62.16 | 62.14 +- 0.108 | 64.17 +- 0.113 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 1k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-1k.pkl) | 25.66 | 25.60 +- 0.071 | 27.26 +- 0.077 | trainval70k | 50,000 | 70,000 |
| stylegan2 5k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-5k.pkl) | 26.60 | 26.64 +- 0.086 | 28.17 +- 0.090 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 5k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-5k.pkl) | 10.45 | 10.45 +- 0.047 | 10.99 +- 0.050 | trainval70k | 50,000 | 70,000 |
| stylegan2 10k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-10k.pkl) | 14.75 | 14.88 +- 0.070 | 16.04 +- 0.078 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 10k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-10k.pkl) | 7.86 | 7.82 +- 0.045 | 8.12 +- 0.044 | trainval70k | 50,000 | 70,000 |
| stylegan2 30k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-ffhq-30k.pkl) | 6.16 | 6.14 +- 0.064 | 6.49 +- 0.068 | trainval70k | 50,000 | 70,000 |
| DiffAugment-stylegan2 30k | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-ffhq-30k.pkl) | 5.05 | 5.07 +- 0.030 | 5.18 +- 0.032 | trainval70k | 50,000 | 70,000 |



<br>

**CIFAR-10**

| Model     | Checkpoint | Reported-FID | Legacy-FID (reproduced)    | Clean-FID  | Reference Split | # generated images used| # reference images used |
| :---:     | :-: | :-:          | :-:          | :-:         | :-: | :-: | :-: |
| stylegan2 10% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10-0.1.pkl) | 36.02 | 36.01 +- 0.161 | 43.64 +- 0.214 | test | 10,000 | 10,000 |
| DiffAugment-stylegan2 10% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10-0.1.pkl) | 14.50 | 14.50 +- 0.121 | 16.94 +- 0.148 | test | 10000 | 10,000 |
| stylegan2 20% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10-0.2.pkl) | 23.08 | 22.98 +- 0.121 | 29.48 +- 0.183 | test | 10,000 | 10,000 |
| DiffAugment-stylegan2 20% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10-0.2.pkl) | 12.15 | 12.13 +- 0.122 | 14.17 +- 0.133 | test | 10,000 | 10,000 |
| stylegan2 100% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/stylegan2-cifar10.pkl) | 11.07 | 11.08 +- 0.080 | 12.94 +- 0.060 | test | 10,000 | 10,000 |
 | DiffAugment-stylegan2 100% | [ckpt](https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-cifar10.pkl) | 9.89 | 9.90 +- 0.100 | 10.87 +- 0.103 | test | 10,000 | 10,000 |
