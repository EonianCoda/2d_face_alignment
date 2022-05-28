import torch
import numpy as np

def NME(pred, gt, average=True) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    else:
        pred = np.array(pred)
    gt = np.array(gt)
    if pred.ndim != gt.ndim:
        raise ValueError("Prediction and ground truth should have same dimensions!")

    dist = np.sqrt(np.sum((pred - gt) ** 2, (-1,-2)))
    if average:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist)
    dist /= 384
    return float(dist)

def heatmap_to_landmark(pred):
    """Convert the model output to keypoints
    """
    pred = pred[-1].detach().cpu()

    lmy = torch.argmax(torch.max(pred, dim=2)[0], dim=2) * 4
    lmx = torch.argmax(torch.max(pred, dim=3)[0], dim=2) * 4
    

    landmark = torch.stack((lmx, lmy), dim=2)
    return landmark
