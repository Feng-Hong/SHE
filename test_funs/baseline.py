import torch
from data.dataloader import get_loader
import numpy as np

class test():
    def __init__(self, config, logger, model, val=False):
        self.config = config
        self.logger = logger
        self.model = model.module
        self.val = val

        # ============================================================================
        # create dataloader
        if val:
            self.logger.info('=====> Create validation dataloader')
            self.loader = get_loader(config, 'test', logger)
        else:
            self.logger.info('=====> Create testing dataloader')
            self.loader = get_loader(config, 'test', logger)

    def run(self, epoch):
        if self.val:
            self.logger.info('=====> Start Baseline Validation at Epoch {}'.format(epoch))
            phase = "val"
        else:
            self.logger.info('=====> Start Baseline Testing')
            phase = "test"

        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.loader):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = np.mean(all_preds == all_labels)
        self.logger.info('=====> {} Accuracy: {:.4f}'.format(phase, acc))

        return acc
    




