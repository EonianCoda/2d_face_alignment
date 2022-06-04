import torch
import numpy as np
from utils.convert_tool import *
import torch.nn as nn
def NME(pred, gt, average=True, return_68=False) -> float:
    pred = to_numpy(pred)
    gt = to_numpy(gt)
    if pred.ndim != gt.ndim:
        raise ValueError("Prediction and ground truth should have same dimensions!")

    dist_68 = np.sqrt(np.sum((pred - gt) ** 2, axis=-1)) / 384 # shpae = ([bs], 68)
    dist = np.mean(dist_68, axis=-1)  # shpae = ([bs])

    if average:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist)

    if not return_68:
        return float(dist)
    else:
        return float(dist), dist_68



def heatmap_to_landmark(heatmap, max_diff=0.26, fix_coord=True):
    """Convert the model output to keypoints
    """
    if isinstance(heatmap, list):
        heatmap = heatmap[-1]
    
    if isinstance(heatmap, torch.Tensor):
        if heatmap.is_cuda:
            heatmap = heatmap.detach().cpu()
    else:
        heatmap = to_tensor(heatmap)

    if len(heatmap.shape) == 4:
        bs, c, h, w = heatmap.shape
        
    else:
        c, h, w = heatmap.shape
        bs = 1
        heatmap = heatmap.unsqueeze(dim=0)


    kernel = nn.Conv2d(68, 68,kernel_size=5 ,stride=1, padding=2,padding_mode="replicate",groups=68 ,bias=False)
    kernel.weight.data[:,0,...] = torch.tensor([[0     , 0.4421, 0.5205, 0.4421, 0],
                                                [0.4421, 0.7214, 0.8494, 0.7214, 0.4421],
                                                [0.5205, 0.8494, 1.0000, 0.8494, 0.5205],
                                                [0.4421, 0.7214, 0.8494, 0.7214, 0.4421],
                                                [0     , 0.4421, 0.5205, 0.4421, 0]])                                 
    kernel.weight.requires_grad = False
    sum_heatmap = heatmap.clone()
    sum_heatmap = kernel(sum_heatmap)

    lmy = torch.argmax(torch.max(sum_heatmap, dim=3)[0], dim=2) 
    lmx = torch.argmax(torch.max(sum_heatmap, dim=2)[0], dim=2)

    # lmy = torch.argmax(torch.max(heatmap, dim=3)[0], dim=2) 
    # lmx = torch.argmax(torch.max(heatmap, dim=2)[0], dim=2)

    
    landmark = torch.stack((lmx, lmy), dim=2)
    if fix_coord:
        offsets = torch.zeros(bs, c, 2).float()
        for batch_i in range(bs):
            for class_i in range(c):
                hm = heatmap[batch_i, class_i,...]
                x, y = landmark[batch_i, class_i, 0], landmark[batch_i, class_i, 1]
                center_prob = hm[y, x]
                if x > 0 and  y > 0 and (x + 1) < w and (y + 1) < h:
                    # left
                    left = hm[y-1 : y+2, max(x-2, 0) : x].mean()
                    # right
                    right = hm[y-1 : y+2, x+1: min(x+3,w)].mean()
                    # top
                    top = hm[max(y-2, 0) : y, x-1 : x+2].mean()
                    # bottom
                    bottom = hm[y+1 : min(y+3,h), x-1 : x+2].mean()
                    # X
                    offsets[batch_i, class_i, 0] = ((right - left) / center_prob) * (1 / max_diff) * 2.0 
                    # Y
                    offsets[batch_i, class_i, 1] = ((bottom - top) / center_prob) * (1 / max_diff) * 2.0

                else:
                    offsets[batch_i, class_i, 0] = 0
        
    landmark = landmark.float()*4
    if fix_coord:
        landmark += offsets
    return landmark
