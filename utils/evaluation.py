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
    dist_68 *= 100
    dist = np.mean(dist_68, axis=-1)  # shpae = ([bs])

    if average:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist)

    if not return_68:
        return float(dist)
    else:
        return float(dist), dist_68

# class Group_estimator(object):
#     def __init__(self):
#         self.groups = {'left_cheek':(0, 8),
#                         'right_cheek':(9, 17),
#                         #'left_eyebrow':(17, 22),
#                         'right_eyebrow':(22, 27)}
#                         #'left_eyelid':(36,42),
#                         #'right_eyelid':(42,48),
#                         #'noise':(27,31),
#                         #'noise_bot':(31,36)}
#                         #'lip':(48,68)}

#     def cal_z(self, pred:torch.Tensor):
#         """
#         Args:
#             pred: shape=(bs, 68, 2)
#         """
#         cal_dist = lambda a,b : torch.sqrt(((a - b) ** 2).sum(dim=-1))

#         bs, num_landmark, _ = pred.shape
#         dists = np.zeros((bs, num_landmark))


#         for batch_i in range(bs):
#             for name, idx in self.groups.items():
#                 target = pred[batch_i, idx[0]:idx[1]]
#                 dists[batch_i, idx[0]:idx[1]] = cal_dist(target.mean(dim=0), target).numpy()

#                 # offset = idx[1] - idx[0]
#                 # dists[batch_i, idx[0]:idx[1]] /= np.maximum(np.abs((np.arange(offset) - ((offset - 1) / 2))), np.ones(offset))
#            # dists[batch_i, 0] *= 0.9

#         return dists

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
    heatmap_temp = heatmap.clone()

    # sum_heatmap = heatmap.sum(dim=0)
    # heatmap = torch.clamp(heatmap, max=1.0)
    heatmap_temp = kernel(heatmap_temp)

    lmy = torch.argmax(torch.max(heatmap_temp, dim=3)[0], dim=2) 
    lmx = torch.argmax(torch.max(heatmap_temp, dim=2)[0], dim=2)

    # lmy = torch.argmax(torch.max(heatmap, dim=3)[0], dim=2) 
    # lmx = torch.argmax(torch.max(heatmap, dim=2)[0], dim=2)

    
    landmark = torch.stack((lmx, lmy), dim=2)
    # fix_outlier = True
    # if fix_outlier:
    #     estimator = Group_estimator()
    #     scores = estimator.cal_z(landmark.float())

    #     offset_ratio = 3
    #     for batch_i in range(len(scores)): 
    #         score = scores[batch_i]
    #         cur_landmark = landmark[batch_i]
    #         revised = []
    #         for name, idxs in estimator.groups.items():
    #             x = score[idxs[0]:idxs[1]]
    #             def cal_quantile(x):
    #                 x1 = np.quantile(x, 0.25)
    #                 x3 = np.quantile(x, 0.75)
    #                 x4 = x3 - x1
    #                 r = 2.2
    #                 upper = x3 + r * x4 
    #                 lower = x3 - r * x4
    #                 return lower, upper
    #             lower, upper = cal_quantile(x)
    #             mask = (x <= upper) & (x >= lower)

    #             # Some outlier
    #             if mask.sum() != (idxs[1] - idxs[0]):
    #                 index = np.arange(idxs[1] - idxs[0]) + idxs[0]
    #                 valid_landmark = cur_landmark[index[mask]].numpy()

    #                 invalid_idx = index[~mask]
    #                 revised_list = np.zeros((len(invalid_idx), 5))

    #                 # X coordinate
    #                 x = valid_landmark[..., 0] 
    #                 revised_list[...,2] = np.min(x)
    #                 revised_list[...,4] = np.max(x)
    #                 # Y coordinate
    #                 y = valid_landmark[..., 1]
    #                 revised_list[...,1] = np.min(y)
    #                 revised_list[...,3] = np.max(y)

    #                 for i, idx in enumerate(invalid_idx):
    #                     revised_list[i, 0] = idx
    #                 revised.append(revised_list)

    #         kernel = nn.Conv2d(1, 1,kernel_size=5 ,stride=1, padding=2,padding_mode="replicate",groups=1 ,bias=False)
    #         kernel.weight.data[:,0,...] = torch.tensor([[0     , 0.4421, 0.5205, 0.4421, 0],
    #                                                     [0.4421, 0.7214, 0.8494, 0.7214, 0.4421],
    #                                                     [0.5205, 0.8494, 1.0000, 0.8494, 0.5205],
    #                                                     [0.4421, 0.7214, 0.8494, 0.7214, 0.4421],
    #                                                     [0     , 0.4421, 0.5205, 0.4421, 0]])                                 
    #         kernel.weight.requires_grad = False

    #         for info in revised:
    #             info = info[0]
    #             landmark_i = int(info[0])
    #             y_min, x_min, y_max, x_max = info[1:].astype(np.int32)

    #             x_offset = (x_max - x_min) // offset_ratio + 1
    #             y_offset = (y_max - y_min) // offset_ratio + 1
    #             y_min = max(y_min - y_offset, 0)
    #             x_min = max(x_min - x_offset, 0)
    #             y_max = min(y_max + y_offset, 96)
    #             x_max = min(x_max + x_offset, 96)

    #             target_heatmap = heatmap[0, landmark_i, y_min:y_max, x_min:x_max]
    #             target_heatmap = kernel(target_heatmap.unsqueeze(dim=0))
    #             y = torch.argmax(torch.max(target_heatmap, dim=2)[0], dim=1) 
    #             x = torch.argmax(torch.max(target_heatmap, dim=1)[0], dim=1)
    #             landmark[batch_i, landmark_i, 0] = x + x_min
    #             landmark[batch_i, landmark_i, 1] = y + y_min


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
                    offsets[batch_i, class_i, 0] = max(min(((right - left) / center_prob), 1.0), -1.0) * (1 / max_diff) * 2.0 
                    # Y
                    offsets[batch_i, class_i, 1] = max(min(((bottom - top) / center_prob), 1.0), -1.0) * (1 / max_diff) * 2.0

                else:
                    offsets[batch_i, class_i, 0] = 0
        
    landmark = landmark.float()*4
    if fix_coord:
        landmark += offsets
    return landmark
