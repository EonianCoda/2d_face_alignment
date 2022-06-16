import cv2
import torch
from utils.convert_tool import to_numpy, is_None
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def read_img(im_path:str):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def draw_point(im, coord:tuple, color:tuple, text:str=None):
    if text != None:
        text_coord = (coord[0] - 5 * len(text), coord[1] - 5)
        cv2.putText(im,
                    text,
                    text_coord,
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color)
    cv2.circle(im, 
                center=coord, 
                radius=3, 
                color=color, 
                thickness=-1)
    return im
def plot_keypoints(im, gt=None, pred=None, show_index:bool=True, show_line:bool=True):
    im = to_numpy(im).copy()

    if is_None(gt) and is_None(pred):
        raise ValueError("Groud truth label and predicting lable are None!")

    if not is_None(gt) and isinstance(gt, torch.Tensor):
        if gt.dim() == 3:
            gt = gt.squeeze(dim=0)
        gt = gt.long().tolist()

    if not is_None(pred) and isinstance(pred, torch.Tensor):
        if pred.dim() == 3:
            pred = pred.squeeze(dim=0)
        pred = pred.long().tolist()
    
    # draw points
    if not is_None(gt):
        for i, (gt_x, gt_y) in enumerate(gt):
            text = str(i+1) if show_index else None
            im = draw_point(im, 
                        text=text,
                        coord=(gt_x, gt_y), 
                        color=(255, 0, 0))
    if not is_None(pred):
        for i, (pred_x, pred_y) in enumerate(pred):
            text = str(i+1) if show_index else None
            im = draw_point(im, 
                        text=text,
                        coord=(pred_x, pred_y), 
                        color=(0, 0, 255))
    # draw lines
    if show_line and not is_None(gt) and not is_None(pred):
        for i, ((gt_x, gt_y), (pred_x, pred_y)) in enumerate(zip(gt, pred)):
            im = cv2.line(im, (gt_x, gt_y), (pred_x, pred_y), color=(0,255,0), thickness=1)

    return im

def plot_loss_68(loss:np.ndarray):
    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel("index of landmark")
    plt.ylabel("average loss")
    plt.show()

class Heatmap_visualizer(object):
    """ref from https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
    """
    @staticmethod
    def get_color_map(color="red"):
        if color == "red":
            cmp = plt.cm.Reds
        elif color == "blue":
            cmp = plt.cm.Blues
        elif color == "green":
            cmp = plt.cm.Greens
        cmp._init()
        cmp._lut[:,-1] = np.linspace(0, 1.0, 255+4)
        return cmp

    def draw_heatmap(self, im, heatmap:torch.Tensor, color="red", ax=None):
        # Processing heatmap
        if heatmap.is_cuda:
            heatmap = heatmap.detach().cpu()
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(dim=0)
        elif heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(dim=0).unsqueeze(dim=0)
        heatmap = F.interpolate(heatmap, scale_factor=(4, 4), mode='nearest')[0]
        heatmap = heatmap.sum(dim=0)
        heatmap = torch.clamp(heatmap, max=1.0)

        im = to_numpy(im)
        if ax == None:
            plt.figure()
            ax = plt.gca()

        #ax.axis('off')
        ax.imshow(im)
        w, h, c = im.shape
        y, x = np.mgrid[0:h, 0:w]
        im = ax.contourf(x, y, heatmap, 5, cmap=self.get_color_map(color))
    