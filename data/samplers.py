import random
import torch
import numpy as np
from torch.utils.data.sampler import Sampler

class DistributionSampler(Sampler):
    def __init__(self, dataset,seed=None):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float() # init weight
        self.epoch = 0
        self.seed = seed


    def __iter__(self):
        self.prob = self.weight / self.weight.sum()
        g = torch.Generator()
        g.manual_seed(self.seed+self.epoch)
        if torch.max(self.weight) == torch.min(self.weight):
            indices = torch.multinomial(self.prob, self.num_samples, replacement=False, generator=g).tolist()
        else:
            indices = torch.multinomial(self.prob, self.num_samples, replacement=True, generator=g).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()

    def set_epoch(self, epoch):
        self.epoch = epoch