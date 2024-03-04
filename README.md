
## SecMI on Latent Diffusion Models

This codebase implements [SecMI](https://arxiv.org/pdf/2302.01316.pdf) on conditioned generation, including fine-tuned Stable Diffusion (SD) and vanilla SD.
This is built on diffuser-0.11.1. Please refer to [here](https://github.com/huggingface/diffusers/tree/v0.11.1) for environment configuration.

We have modified the following files in order to perform SecMI:
```shell
# Implement reverse in DDIM
./src/diffusers/schedulers/scheduling_ddim.py
# Save reverse and denoising intermediate results
./src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
```

### Dataset
Please download datasets from [here](https://drexel0-my.sharepoint.com/:u:/g/personal/jd3734_drexel_edu/EeEwxOQ-5cZEnf534S6WRkQBOcvbAtfmuV-h5UjyIF8YxQ?e=JYAHfo) and `unzip` it.

There are three datasets included: [Pokemon](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions), [Laion](https://laion.ai/blog/laion-5b/) (2.5k), and [coco2017val](https://cocodataset.org/#home) (2.5k).

### SecMI w/ Stable Diffusion fine-tuned over the Pokemon dataset

#### Checkpoint
Please download the Pokemon-fine-tuned SD from [here](https://drexel0-my.sharepoint.com/:u:/g/personal/jd3734_drexel_edu/EYX4y5AgG9ZMjbSUvd0Oc3MBRaSBmZTqZjAVkOoG6kjIEw?e=M66BHj) and `unzip` it.

#### Script
Please refer to the following script or run `sh scripts/secmi_ldm_pokemon.sh`
```shell
CUDA_VISIBLE_DEVICES=0 python -m src.mia.secmi \
--dataset pokemon \
--dataset-root /path/to/datasets \
--ckpt-path /path/to/sd-pokemon-checkpoint
```

### SecMI w/ vanilla Stable Diffusion over Laion (as member) and coco2017val (as non-member)

#### Script
Please refer to the following script or run `sh scripts/secmi_sd_laion.sh`
```shell
CUDA_VISIBLE_DEVICES=0 python -m src.mia.secmi \
--dataset laion \
--dataset-root /path/to/datasets \
--ckpt-path runwayml/stable-diffusion-v1-5
```

### Reference
Please cite our paper if you feel this is helpful:
```
@InProceedings{duan2023are,
  title = {Are Diffusion Models Vulnerable to Membership Inference Attacks?},
  author = {Duan, Jinhao and Kong, Fei and Wang, Shiqi and Shi, Xiaoshuang and Xu, Kaidi},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {8717--8730},
  year = {2023}
}
```