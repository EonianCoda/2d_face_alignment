from re import X
import torch
import numpy as np

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    else:
        return np.array(x)

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.tensor(x)