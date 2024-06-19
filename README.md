<h1 align="center">On Harmonizing Implicit Subpopulations</h1>

<p align="center">
    <!-- <a href="https://arxiv.org/abs/2406.04872"><img src="https://img.shields.io/badge/arXiv-2406.04872-b31b1b.svg" alt="Paper"></a> -->
    <a href="https://openreview.net/pdf?id=3GurO0kRue"><img src="https://img.shields.io/badge/OpenReview-ICLR'24-blue" alt="Paper"></a>
    <a href="https://github.com/MediaBrain-SJTU/SHE"><img src="https://img.shields.io/badge/Github-RECORDS-brightgreen?logo=github" alt="Github"></a>
    <!-- <a href="https://iclr.cc/media/iclr-2023/Slides/11305.pdf"> <img src="https://img.shields.io/badge/Slides (5 min)-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
    <a href="https://iclr.cc/media/PosterPDFs/ICLR%202024/19522.png?t=1715875121.2257736"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>

by Feng Hong, Jiangchao Yao, Yueming Lyu, Zhihan Zhou, Ivor Tsang, Ya Zhang, and Yanfeng Wang at SJTU, Shanghai AI Lab, A*STAR, and NTU.

International Conference on Learning Representations (ICLR), 2024.

This repository is the official Pytorch implementation of SHE.

## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{
hong2024on,
title={On Harmonizing Implicit Subpopulations},
author={Feng Hong and Jiangchao Yao and Yueming Lyu and Zhihan Zhou and Ivor Tsang and Ya Zhang and Yanfeng Wang},
booktitle={ICLR},
year={2024}
}
```

## Environment

The project is tested under the following environment settings:
- OS: Ubuntu 18.04.5
- GPU: NVIDIA GeForce RTX 3090s
- Python: 3.7.10
- PyTorch: 1.13.1
- Torchvision: 0.8.2
- Cudatoolkit: 11.0.221
- Numpy (1.21.2) 

## Content
- ```./utils```: logger, optimizers, loss functions, and misc
- ```./models```: backbone models
- ```./data```: datasets and dataloaders
- ```./cfg```: config files
- ```./exp```: path to store experiment logs and checkpoints
- ```./train_funs```: train functions
- ```./test_funs```: test functions
- ```main.py```: main function

## Usage

**Run SHE on COCO**

```shell
# COCO
PORT=$[$RANDOM + 10000]
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/SHE.yaml --phase train --seed 0 --port $PORT
```

## Extensions

**Implement Your Own Model**

- Add your model to ```./models``` and import the model in ```./models/__init__.py```.

**Implement Your Own Method**

- Add your method to ```./train_funs``` and import the method in ```./train_funs/__init__.py```.

**Implement Other Datasets**

- Add raw data to ```./[_data_name]```.
- Create subpopulation-imabalnced version of the dataset in ```./data```.
- Create dataloader in ```./data/dataloader.py```.

## Contact
If you have any problem with this code, please feel free to contact **feng.hong@sjtu.edu.cn**.
