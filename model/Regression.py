from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0

import torch.nn as nn
import copy

class RegressionModel(nn.Module):
    def __init__(self, backbone="mobilenet_v2", num_landmark=68):
        super(RegressionModel, self).__init__()
        self.num_landmark = num_landmark
        if backbone == "mobilenet_v2":
            model = mobilenet_v2(pretrained = True)
            self.features = copy.deepcopy(model._modules['features'])
            del model
            self.regressor =  nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(1280, num_landmark * 2)
                                            )
        elif backbone == "efficientnet_b0":
            model = efficientnet_b0(pretrained = True)
            self.features = copy.deepcopy(model._modules['features'])
            del model
            self.regressor =  nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(1280, num_landmark * 2)
                                            )
        self._weight_init()
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        bs, c = x.shape
        x = x.reshape((bs, self.num_landmark, 2))
        return x