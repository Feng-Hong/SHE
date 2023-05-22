
import torch.nn as nn
import torchvision.models as models

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