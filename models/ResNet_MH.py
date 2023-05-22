import torch.nn as nn
import torchvision.models as models
import math
import torch

class Classifier_multi_head(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, num_head=2):
        super(Classifier_multi_head, self).__init__()

        # classifier weights
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        # x: N, dim
        x_list = torch.split(x, self.head_dim, dim=1)
        w_list = torch.split(self.weight, self.head_dim, dim=1)
        out = []

        # for x_i, w_i in zip(x_list, w_list):
        #     y_i = torch.mm(x_i, w_i.t())
        #     out.append(y_i)
        for i in range(self.num_head):
            out.append(torch.mm(x_list[i], w_list[i].t()))
            
        return out
    
class ResNet_MH(nn.Module):
    def __init__(self,m_type='resnet101', num_classes=1000, num_head=2, pretrained = False):
        super(ResNet_MH, self).__init__()
        if m_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif m_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif m_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif m_type == 'resnext50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
        elif m_type == 'resnext101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.fc = Classifier_multi_head(self.resnet.fc.in_features, num_classes, num_head)
        self.resnet.fc = nn.Identity()

    def forward(self, x, feature=False):
        feat = self.resnet(x)
        out = self.fc(feat)
        if feature:
            return out, feat
        elif self.training:
            return out
        else:
            return torch.logsumexp(torch.stack(out), dim=0)