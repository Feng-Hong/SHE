import random
import numpy as np
import torch
import torch.utils.data as data
from .coco import coco_imbatt_balcls, coco_imbatt_balcls_transform
from .samplers import DistributionSampler

def get_loader(config, phase, logger):
    # set random seed to config['seed']['dataset']
    random.seed(config['seed']['dataset'])
    np.random.seed(config['seed']['dataset'])
    torch.manual_seed(config['seed']['dataset'])
    torch.cuda.manual_seed(config['seed']['dataset'])
    torch.cuda.manual_seed_all(config['seed']['dataset'])

    if config['dataset']['name'] == 'coco_imbatt_balcls':
        #(phase, rgb_mean, rgb_std, rand_aug)
        config['dataset']['RandomResizedCrop'] = None if 'RandomResizedCrop' not in config['dataset'] else config['dataset']['RandomResizedCrop']
        config['dataset']['test_Resize'] = None if 'test_Resize' not in config['dataset'] else config['dataset']['test_Resize']
        config['dataset']['test_CenterCrop'] = None if 'test_CenterCrop' not in config['dataset'] else config['dataset']['test_CenterCrop']
        transform = coco_imbatt_balcls_transform(phase, config['dataset']['rgb_mean'], config['dataset']['rgb_std'], config['dataset']['rand_aug'], config['dataset']['RandomResizedCrop'], config['dataset']['test_Resize'], config['dataset']['test_CenterCrop'])
        dataset = coco_imbatt_balcls(config['dataset']['data_path'], phase, logger, transform)
    else:
        raise NotImplementedError
    
    # data sampler
    sampler_type = config['sampler']
    if phase != 'train':
        sampler_type = 'default'

    if sampler_type == 'default':
        # ddp
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if phase == 'train' else None
        # DistributedSampler will shuffle the dataset by default
        shuffle = False # for validation and test
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=config['training_opt']['batch_size'] if phase == 'train' else config['training_opt']['batch_size'],
            shuffle=shuffle,
            num_workers=config['training_opt']['data_workers'],
            pin_memory=True,
            sampler=sampler,
            drop_last=True if phase == 'train' else False
        )
    elif sampler_type == 'DistributionSampler':
        logger.info('======> Sampler Type {}, Sampler Number {}'.format(sampler_type, config['num_sampler']))
        loader = []
        num_sampler = config['num_sampler']
        batch_size =  config['training_opt']['batch_size']
        if num_sampler>1:
            if config['batch_split']:
                batch_size = batch_size // num_sampler
        for i in range(num_sampler):
            # sampler = DistributionSampler_multi(dataset=dataset,rank=i, num_sampler=num_sampler,seed=config['seed']['dataset'])
            sampler = DistributionSampler(dataset=dataset,seed=config['seed']['dataset'])
            loader.append(data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config['training_opt']['data_workers'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True if phase == 'train' else False
            ))
        if len(loader) == 1:
            loader = loader[0]

    return loader
