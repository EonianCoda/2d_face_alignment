from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch
import copy

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        model = mobilenet_v2(pretrained = True)
        self.feat_extractor = copy.deepcopy(model._modules['features'])
        del model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(1280, 68)
    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x