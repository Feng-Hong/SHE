import torch.nn as nn
import torch
import models
from data.dataloader import get_loader
from utils.optimizer import create_optimizer,create_scheduler
import test_funs
from utils.loss import create_loss
import os
# dist
import torch.distributed as dist
import numpy as np

class train():
    def __init__(self, config, logger, eval=False):
        # ============================================================================
        # create model
        model_type = config['networks']['type']
        model_args = config['networks']['params']
        model = getattr(models, model_type)(**model_args)
        if config['distributed']:
            if config['gpu'] is not None:
                torch.cuda.set_device(config['gpu'])
                model.cuda(config['gpu'])
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']], find_unused_parameters=False)
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        else:
            raise NotImplementedError

        self.config = config
        self.logger = logger
        self.model = model
        self.eval = eval
        self.training_opt = config['training_opt']
        self.optimizer = create_optimizer(model, logger, config)
        self.scheduler = create_scheduler(self.optimizer, logger, config)
        try:
            self.num_head = config['networks']['params']['num_head']
        except:
            self.num_head = config['networks']['params']['num_experts']
        self.num_classes = config['networks']['params']['num_classes']

        # ============================================================================
        # create dataloader
        self.train_loader = get_loader(config, 'train', logger)
        self.num_samples = len(self.train_loader.dataset)
        # self.weight_per_sample = torch.ones((self.num_samples, self.num_head), dtype=torch.float)
        # self.weight_per_sample = self.weight_per_sample / self.num_head
        # self.weight_per_sample = nn.Parameter(torch.ones((self.num_samples, self.num_head), dtype=torch.float), requires_grad=True)
        self.weight_per_sample = nn.ParameterList([nn.Parameter(torch.zeros(self.num_head, dtype=torch.float), requires_grad=True) for i in range(self.num_samples)]).cuda()
        self.gt_per_sample = torch.zeros(self.num_samples, dtype=torch.long).cuda()
        self.gt_per_sample = self.gt_per_sample - 1
        self.update_gt = True

        self.resample_weight = torch.ones(self.num_samples, dtype=torch.float)
        # self.weight_optim = torch.optim.SGD([self.weight_per_sample], lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.weight_optim = torch.optim.SGD(self.weight_per_sample, lr=self.config['algorithm_opt']['weight_lr'][0], momentum=0.9, weight_decay=0.0005)
        self.weight_optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.weight_optim, T_max=200, eta_min=self.config['algorithm_opt']['weight_lr'][1])
        # ============================================================================
        # create loss
        self.loss = create_loss(config, logger)

        self.testing = None
        if self.eval:
            if not config['multiprocessing_distributed'] or \
                (config['multiprocessing_distributed'] and config['rank'] % config['ngpus_per_node'] == 0):
                self.testing = getattr(test_funs, config['testing_opt']['type'])(config, logger, model, val=True)
        self.weight_per_sample.requires_grad_(False)

    def get_weight(self, index):
        # weight = self.weight_per_sample[index]
        self.weight_per_sample.requires_grad_(False)
        weight = torch.stack([self.weight_per_sample[i].requires_grad_(True) for i in index])
        # softmax
        weight = torch.softmax(weight, dim=1) # (batch, num_head)
        return weight
    
    def _entropy_per_head(self):
        # weight = self.weight_per_sample
        weight = torch.stack([self.weight_per_sample[i] for i in range(self.num_samples)])
        weight = torch.softmax(weight, dim=1) # (num_samples, num_head)
        sum_weight_per_head_per_class = torch.zeros(self.num_classes, self.num_head).cuda()
        for i in range(self.num_classes):
            sum_weight_per_head_per_class[i] = weight[self.gt_per_sample == i].sum(dim=0) + 1e-8
        self.sum_weight_per_head_per_class = sum_weight_per_head_per_class
        sum_weight_per_head_per_class_head_norm = sum_weight_per_head_per_class / sum_weight_per_head_per_class.sum(dim=1, keepdim=True)# (num_classes, num_head), norm to 1 for each head
        # entropy
        entropy_per_head = -torch.sum(sum_weight_per_head_per_class_head_norm * torch.log(sum_weight_per_head_per_class_head_norm + 1e-8), dim=0) # (num_head)
        # weighted sum
        weight_per_head = torch.sum(sum_weight_per_head_per_class, dim=0) # (num_head)
        weight_per_head = weight_per_head / weight_per_head.sum()
        entropy = torch.sum(entropy_per_head * weight_per_head)
        return entropy

    def run(self):
        # ============================================================================
        # train
        best_acc = 0.0
        if self.testing is not None:
            self.logger.info('=====> Start training')
            
        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            if self.config['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            # train
            if self.testing is not None:
                self.logger.info('=====> Epoch: {}'.format(epoch))
            
            self.model.train()
            total_batch = len(self.train_loader)

            for i, data in enumerate(self.train_loader):
                # get data
                img, label, attribute, index = data
                # to device
                if self.config['distributed']:
                    img = img.cuda(self.config['gpu'], non_blocking=True)
                    label = label.cuda(self.config['gpu'], non_blocking=True)
                    # attribute = attribute.cuda(self.config['gpu'], non_blocking=True)
                # forward
                # predictions, all_logits = self.model(img)
                if self.update_gt:
                    if torch.sum(self.gt_per_sample < 0) > 0:
                        self.gt_per_sample[index] = label
                    else:
                        self.update_gt = False
                all_logits = self.model(img)

                

                if epoch < self.config['algorithm_opt']['warmup_epoch'] or self.update_gt:
                    weight = torch.ones((label.shape[0], self.num_head), dtype=torch.float).cuda()
                    weight = weight / self.num_head
                    entropy_per_head = 0.0
                else:
                    weight = self.get_weight(index)
                    entropy_per_head = -self._entropy_per_head() #
                    # entropy_per_head = 0.0
                
                # loss
                loss = self.loss(all_logits, label, weight.cuda())

                loss = loss + entropy_per_head * self.config['algorithm_opt']['entropy_weight']

                # backward
                self.optimizer.zero_grad()
                self.weight_optim.zero_grad(set_to_none=True)
                loss.backward()
                if (epoch >= self.config['algorithm_opt']['warmup_epoch']) and (self.update_gt is False):
                    if self.testing is not None:
                        if i % self.config['logger_opt']['print_iter'] == 0:
                            self.logger.info(self.weight_per_sample[index[0]].grad)
                self.optimizer.step()
                self.weight_optim.step()

                # print
                if self.testing is not None:
                    if i % self.config['logger_opt']['print_iter'] == 0:
                        # train acc
                        predictions = sum(all_logits) if isinstance(all_logits, list) else all_logits
                        _, predicted = torch.max(predictions.data, 1)
                        total = label.size(0)
                        correct = (predicted == label).sum().item()
                        train_acc = correct / total
                        log_entropy = entropy_per_head if (epoch < self.config['algorithm_opt']['warmup_epoch'] or self.update_gt) else entropy_per_head.item() 
                        # log_entropy = entropy_per_head

                        self.logger.info(f'Index: {index[0]}, weight: {weight[0]}')
                        self.logger.info(f'Epoch: {epoch}/{self.training_opt["num_epochs"]}, Iter: {i}/{total_batch}, Loss: {loss.item():.4f}, entropy: {log_entropy}, Train acc: {train_acc:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # scheduler
            self.scheduler.step()
            self.weight_optim_scheduler.step()



            dist.barrier()
            # validation
            if self.testing is not None:
                self.logger.wandb_log({'loss': loss.item(),'entropy': log_entropy, 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], self.training_opt['loss']: loss.item()})

                if self.eval:
                    val_acc = self.testing.run(epoch)
                    self.logger.wandb_log({'val_acc': val_acc, 'epoch': epoch})
                else:
                    val_acc = 0.0
                # print best acc
                self.logger.info(f'=====> Best val acc: {max(best_acc, val_acc):.4f}, Current val acc: {val_acc:.4f}')
                # save model
                self.logger.info('=====> Save model')
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint = {
                        'epoch': epoch,
                        'model_type': self.config['networks']['type'],
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    model_name = f'checkpoint_best.pth'
                    model_path = os.path.join(self.config['output_dir'], model_name)
                    torch.save(checkpoint, model_path)
                    self.logger.info('=====> Save model to {}'.format(model_path))
                else:
                    checkpoint = {
                        'epoch': epoch,
                        'model_type': self.config['networks']['type'],
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    model_name = f'checkpoint.pth'
                    model_path = os.path.join(self.config['output_dir'], model_name)
                    torch.save(checkpoint, model_path)
                    self.logger.info('=====> Save model to {}'.format(model_path))
                
                
        # save self.weight_per_sample to file
        torch.save(self.weight_per_sample.cpu(), os.path.join(self.config['output_dir'], 'weight_per_sample.pth'))

        if self.testing is not None:
            self.logger.info('=====> Finish training')
            self.logger.info('=====> Best val acc: {:.4f}'.format(best_acc))

