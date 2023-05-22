# Latent Subpopulation Rebalancing

This is the source code of LSR (NeurIPS 2023 Submission). Please do not distribute.

## Environment

The project is tested under the following environment settings:
- OS: Ubuntu 18.04.5
- GPU: NVIDIA GeForce RTX 3090
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

**Run LSR on COCO**

```shell
# COCO
PORT=$[$RANDOM + 10000]
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/LSR.yaml --phase train --seed 0 --port $PORT
```

## Extensions

**Implement Your Own Model**

- Add your model to ```./models``` and import the model in ```./models/__init__.py```.

**Implement Other Datasets**

- Add raw data to ```./[_data_name]```.
- Create subpopulation-imabalnced version of the dataset in ```./data```.
- Create dataloader in ```./data/dataloader.py```.