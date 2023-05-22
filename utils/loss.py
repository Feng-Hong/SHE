import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

def create_loss(config,logger):
    training_opt = config['training_opt']
    if training_opt['loss'] == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    elif training_opt['loss'] == 'MHLoss':
        loss = MHLoss(**training_opt["loss_params"])
    else:
        raise NotImplementedError
    return loss

class MHLoss(nn.Module):
    def __init__(self, diversity_weight = -0.2, diversity_temperature = 1.0, diversity_loss_type = 'l2'):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.diversity_temperature = diversity_temperature
        self.diversity_loss_type = diversity_loss_type
        

    def forward(self, all_logits, target, weight=None):
        # if all_logits is a list
        loss = 0
        if isinstance(all_logits, list):
            pred = sum(all_logits) / len(all_logits)
        else:
            pred = all_logits
        if self.diversity_weight != 0:
            with torch.no_grad():
                mean_output_dist = F.softmax(pred / self.diversity_temperature, dim=1)
            for logits_item in all_logits:
                # output_dist = F.log_softmax(logits_item / self.diversity_temperature, dim=1)
                # loss += self.diversity_weight * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
                # l2 loss bw output_dist and mean_output_dist
                output_dist = F.softmax(logits_item / self.diversity_temperature, dim=1)
                if self.diversity_loss_type == 'l2':
                    loss += self.diversity_weight * F.mse_loss(output_dist, mean_output_dist)
                elif self.diversity_loss_type == 'kl':
                    loss += self.diversity_weight * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
                else:
                    raise NotImplementedError
        for i, logits_item in enumerate(all_logits):
            if weight is not None:
                loss += (weight[:, i] * F.cross_entropy(logits_item, target, reduction='none')).mean() # * len(all_logits)
            else:
                loss += F.cross_entropy(logits_item, target) / len(all_logits)
            # loss += F.cross_entropy(logits_item, target)
        # loss = loss / len(all_logits)
        # loss += F.cross_entropy(pred, target)
        return loss
    
