import string
from datetime import datetime
import secrets
def random_str(num):
    salt = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(num))

    return salt
    
def get_date():

    now = datetime.now()
    return str(now.strftime("20%y_%h_%d"))

def re_nest_configs(config_dict):
    flattened_params = [key for key in config_dict.keys() if '.' in key]
    for param in flattened_params:
        value = config_dict._items.pop(param)
        # value = config_dict[param]
        # del config_dict[param] 
        param_levels = param.split('.')
        parent = config_dict._items
        for level in param_levels:
            if isinstance(parent[level], dict):
                parent = parent[level]
            else:
                parent[level] = value

    if 'sweep_config' in config_dict.keys():
        config_dict._items.pop("sweep_config")

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)