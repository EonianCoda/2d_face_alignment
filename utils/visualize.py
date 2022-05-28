import cv2
import torch
import os
import numpy as np
def read_img(im_path:str):
    return cv2.imread(os.path.join("./data/val", im_path))

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

def plot_keypoints(im:np.ndarray, gt:torch.Tensor, pred:torch.Tensor, show_index:bool=True, show_line:bool=True):
    # draw points
    for i, ((gt_x, gt_y), (pred_x, pred_y)) in enumerate(zip(gt, pred)):
        text = str(i+1) if show_index else None
        draw_point(im, 
                    text=text,
                    coord=(gt_x, gt_y), 
                    color=(255, 0, 0))
        draw_point(im, 
                    text=text, 
                    coord=(pred_x, pred_y), 
                    color=(0, 0, 255))
        if show_line:
            cv2.line(im, (gt_x, gt_y), (pred_x, pred_y), color=(0,255,0), thickness=1)

    return im