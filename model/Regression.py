from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0

import torch.nn as nn
import torch
import copy

class RegressionModel(nn.Module):
    def __init__(self, backbone="mobilenet_v2", num_landmark=68):
        super(RegressionModel, self).__init__()
        if backbone == "mobilenet_v2":
            model = mobilenet_v2(pretrained = True)
            self.features = copy.deepcopy(model._modules['features'])
            del model
            self.regressor =  nn.Sequential(nn.AdaptiveAvgPool2d(1, 1),
                                            nn.Flatten(),
                                            nn.Linear(1280, num_landmark)
                                            )
        elif backbone == "efficientnet_b0":
            model = efficientnet_b0(pretrained = True)
            self.features = copy.deepcopy(model._modules['features'])
            del model
            self.regressor =  nn.Sequential(nn.AdaptiveAvgPool2d(1, 1),
                                            nn.Flatten(),
                                            nn.Linear(1280, num_landmark)
                                            )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x