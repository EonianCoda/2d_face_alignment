import torch.nn as nn
import torch

class Weighted_L2(nn.Module):
    def __init__(self,reduction="mean", weight:float=10.0):
        super(Weighted_L2, self).__init__()
        self.weight = weight
        self.reduction = reduction
    def forward(self, pred:torch.Tensor, target:torch.Tensor, weight_map:torch.Tensor):
        w = self.weight * weight_map + 1
        if self.reduction == "sum":
            return (((target - pred) ** 2) * w).sum()
        elif self.reduction == "mean":
            return (((target - pred) ** 2) * w).mean()


