import torch.nn as nn
import torch
import math

class Wing_Loss(nn.Module):
    """ref from "https://github.com/protossw512/AdaptiveWingLoss"
    """
    def __init__(self, omega=10, epsilon=2):
        super(Wing_Loss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
    
    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        delta_y = (target - pred).abs()
        # When |X| < omega
        delta_y1 = delta_y[delta_y < self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        # otherwise
        delta_y2 = delta_y[delta_y >= self.omega]
        loss2 = delta_y2 - self.C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class Adaptive_Wing_Loss(nn.Module):
    """ref from "https://github.com/protossw512/AdaptiveWingLoss"
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(Adaptive_Wing_Loss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.weight = 10
    def forward(self, pred, target, weight_map):
        """
        Args:
            pred: shape=(B, N, H, W)
            target: shape=(B, N, H, W)
        """
        w = self.weight * (weight_map + 1)
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        mask1 = delta_y < self.theta
        mask2 = delta_y >= self.theta
        
        delta_y1 = delta_y[mask1]
        delta_y2 = delta_y[mask2]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)


        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        loss1 *= w[mask1]
        loss2 *= w[mask2]
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
