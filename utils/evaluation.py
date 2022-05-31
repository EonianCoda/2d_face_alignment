from dis import dis
import torch
import numpy as np

def NME(pred, gt, average=True, return_68=False) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    else:
        pred = np.array(pred)
    gt = np.array(gt)
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

def heatmap_to_landmark(heatmap):
    """Convert the model output to keypoints
    """
    heatmap = heatmap[-1].detach().cpu()

    lmy = torch.argmax(torch.max(heatmap, dim=3)[0], dim=2) 
    lmx = torch.argmax(torch.max(heatmap, dim=2)[0], dim=2)
    
    landmark = torch.stack((lmx, lmy), dim=2)
    
    bs, c, h, w = heatmap.shape
    offsets = torch.zeros(bs, c, 2).float()
    offset_n = 2

    for batch_i in range(bs):
        for class_i in range(c):
            hm = heatmap[batch_i, class_i,...]
            x, y = landmark[batch_i, class_i, 0], landmark[batch_i, class_i, 1]
            center_prob = hm[y, x]

            if (x - 1) > 0 and  (y - 1) > 0 and (x + 1) < w and (y + 1) < h:
                diff_percent_x =(hm[y, x + 1] - hm[y, x - 1]) / center_prob
                diff_percent_y =(hm[y + 1, x] - hm[y - 1, x]) / center_prob
                offsets[batch_i, class_i, 0] = (diff_percent_x + 1) * offset_n
                offsets[batch_i, class_i, 0] = (diff_percent_y + 1) * offset_n
            else:
                offsets[batch_i, class_i, 0] = offset_n
        

    landmark = landmark.float()*4
    landmark += offsets

    return landmark
# def heatmap_to_landmark(pred):
#     """Convert the model output to keypoints
#     """
#     pred = pred[-1].detach().cpu()

#     lmy = torch.argmax(torch.max(pred, dim=2)[0], dim=2) * 4
#     lmx = torch.argmax(torch.max(pred, dim=3)[0], dim=2) * 4
    

#     landmark = torch.stack((lmx, lmy), dim=2)
#     return landmark
