import torch
from torch.utils.data import Dataset
from utils.transform import get_transform
import numpy as np
from platform import python_version
from PIL import Image
import os
import random


def process_label(origin_label:np.ndarray):
    # Wrong Label
    if (origin_label >= 384).any() or (origin_label < 0).any():
        return False, None

    label = origin_label.copy()
    label = np.round(label * 0.25)

    same_count = 0
    while True:
        flag = True
        idxs = np.lexsort((label[:,1], label[:,0]))
        for i in range(len(idxs) - 1):
            cur_idx, next_idx = idxs[i], idxs[i + 1]
            # If there are keypoint having the same coordinate,
            # then change one of the point's y.
            if (label[cur_idx] == label[next_idx]).all():
                flag = False
                same_count += 1
                if same_count == 15:
                    return False, None
                if origin_label[cur_idx][1] > origin_label[next_idx][1]:
                    label[cur_idx][1] += 1
                else:
                    label[cur_idx][1] -= 1
                break
        if flag:
            break
    # Wrong label
    if (label >= 384).any() or (label < 0).any():
        return False, None
    label = label.astype(np.int32)
    return True, label

def process_annot(annot_path:str):
    """Read the annot file and process label(e.g. discard wrong label)
    """

    # If python verions < 3.8.0, then use pickle5
    py_version = python_version()
    py_version = int(''.join(py_version.split('.')[:2]))
    if py_version < 38:
        import pickle5 as pickle
    else:
        import pickle
    
    images, labels = pickle.load(open(annot_path, 'rb'))
    
    valid_imgs = []
    valid_labels = []
    gt_labels = []
    for img, label in zip(images, labels):
        result = process_label(label)
        if result[0]:
            valid_imgs.append(img)
            valid_labels.append(result[1])
            gt_labels.append(label)
    
    valid_labels = np.stack(valid_labels)
    gt_labels = np.stack(gt_labels)

    return valid_imgs, valid_labels, gt_labels


def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8, use_image_ratio=1.0):
    """Get training set and valiating set
    Args:
        data_root: the data root for images
        annot_path: thh path of the annotation file
        train_size: the size ratio of train:val
        use_image_ratio: how many images to use in training and validation
    """
    images, labels, gt_labels = process_annot(annot_path)
    

    # Split train/val set
    idxs = [i for i in range(int(len(images) * use_image_ratio))]
    random.shuffle(idxs)

    # Training set
    train_idxs = idxs[: int(len(idxs)*train_size)]
    train_images = [images[i] for i in train_idxs]
    train_labels = labels[train_idxs]
    train_gt_labels = gt_labels[train_idxs]

    # Validation set
    val_idxs = idxs[int(len(idxs)*train_size): ]
    val_images = [images[i] for i in val_idxs]
    val_labels = labels[val_idxs]
    val_gt_labels = gt_labels[val_idxs]

    train_dataset = FaceSynthetics(data_root, train_images, train_labels, train_gt_labels, get_transform('train'))
    val_dataset = FaceSynthetics(data_root, val_images, val_labels, val_gt_labels, get_transform('val'))
    return train_dataset, val_dataset

class FaceSynthetics(Dataset):
    def __init__(self, data_root:str, images:list, labels:np.ndarray, gt_labels:np.ndarray, transform=None, heatmap_size=96) -> None:
        super(FaceSynthetics, self).__init__()
        self.data_root = data_root
        self.images = images
        self.labels= labels
        self.gt_labels = torch.tensor(gt_labels)
        self.transform = transform 
        self.heatmap_size = heatmap_size
        self.num_classes = len(self.labels[0])

    def gen_heatmap(self, label):
        heatmap = np.zeros((self.num_classes, self.heatmap_size, self.heatmap_size))
        for i, (x, y) in enumerate(label):
            heatmap[i, y, x] = 1
        return torch.tensor(heatmap).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int):
        # Read imagee
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im = self.transform(im)
        # training label
        label = self.gen_heatmap(self.labels[idx])
        gt_label = self.gt_labels[idx]
        return im, label, gt_label