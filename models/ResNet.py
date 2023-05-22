
import torch.nn as nn
import torchvision.models as models


# def create_model(m_type='resnet101',num_classes=1000):
#     # create various resnet models
#     if m_type == 'resnet18':
#         model = models.resnet18(pretrained=False) # feature dim = 512
#     elif m_type == 'resnet50':
#         model = models.resnet50(pretrained=False) # feature dim = 2048
#     elif m_type == 'resnet101':
#         model = models.resnet101(pretrained=False) # feature dim = 2048
#     elif m_type == 'resnext50':
#         model = models.resnext50_32x4d(pretrained=False) # feature dim = 2048
#     elif m_type == 'resnext101':
#         model = models.resnext101_32x8d(pretrained=False) # feature dim = 2048
#     else:
#         raise ValueError('Wrong Model Type')
#     # model.fc = nn.ReLU()
#     # model.fc = nn.Identity()
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

class create_model(nn.Module):
    def __init__(self, m_type='resnet101',num_classes=1000, pretrained = False):
        super(create_model, self).__init__()
        # create various resnet models
        if m_type == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif m_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif m_type == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif m_type == 'resnext50':
            self.model = models.resnext50_32x4d(pretrained=pretrained)
        elif m_type == 'resnext101':
            self.model = models.resnext101_32x8d(pretrained=pretrained)
        else:
            raise ValueError('Wrong Model Type')
        
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Identity()
    def forward(self, x, feature=False):
        feat= self.model(x)
        out = self.fc(feat)
        if feature:
            return out, feat
        else:
            return out


if __name__ == '__main__':
    model = models.resnet18(pretrained=False)
    print(model.fc.in_features)
    model = models.resnet50(pretrained=False)
    print(model.fc.in_features)
    model = models.resnet101(pretrained=False)
    print(model.fc.in_features)
    model = models.resnext50_32x4d(pretrained=False)
    print(model.fc.in_features)
    model = models.resnext101_32x8d(pretrained=False)
    print(model.fc.in_features)