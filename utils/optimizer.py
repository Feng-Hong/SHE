import torch.optim as optim
import torch

def create_optimizer(model, logger, config, other_models=None):
    training_opt = config['training_opt']
    lr = training_opt['optim_params']['lr']
    weight_decay = training_opt['optim_params']['weight_decay']
    if training_opt['optimizer'] == 'SGD':
        momentum = training_opt['optim_params']['momentum']
    else:
        momentum = None
    
    if training_opt['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif training_opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def create_scheduler(optimizer, logger, config):
    training_opt = config['training_opt']
    scheduler_params = training_opt['scheduler_params']
    if training_opt['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=scheduler_params['gamma'], step_size=scheduler_params['step_size'])
    elif training_opt['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
    elif training_opt['scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_params['milestones'], gamma=scheduler_params['gamma'])
    else:
        raise NotImplementedError
    return scheduler
