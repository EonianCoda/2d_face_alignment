from pyexpat import model
import torch
from torch.utils.data import Dataset
from utils.transform import get_transform
import numpy as np
from platform import python_version
from PIL import Image
import os
import random
import math
import copy
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
    if (label >= 96).any() or (label < 0).any():
        return False, None
    label = label.astype(np.int32)
    return True, label

def process_annot(annot_path:str, model_type:str):
    """Read the annot file and process label(e.g. discard wrong label)
    """
    path = os.path.dirname(annot_path)
    file_name = os.path.basename(annot_path).split('.')[0]
    cached_file = f'cached_{file_name}_{model_type}.pkl'
    cached_file = os.path.join(path, cached_file)
    

    # If python verions < 3.8.0, then use pickle5
    py_version = python_version()
    py_version = int(''.join(py_version.split('.')[:2]))
    if py_version < 38:
        import pickle5 as pickle
    else:
        import pickle

    if model_type == "regressor":
        images, labels = pickle.load(open(annot_path, 'rb'))
        return images, labels, copy.deepcopy(labels)
    
    ### The code below are for model_type == "classifier" ###

    # If cached file exists, then load it.
    if os.path.isfile(cached_file):
        return pickle.load(open(cached_file, 'rb'))
    else:
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
    # Save cached file
    if not os.path.isfile(cached_file):
        pickle.dump((valid_imgs, valid_labels, gt_labels), open(cached_file, 'wb'))

    return valid_imgs, valid_labels, gt_labels

def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8, use_image_ratio=1.0, model_type="classifier",transform:dict=None):
    """Get training set and valiating set
    Args:
        data_root: the data root for images
        annot_path: thh path of the annotation file
        train_size: the size ratio of train:val
        use_image_ratio: how many images to use in training and validation
    """
    images, labels, gt_labels = process_annot(annot_path, model_type=model_type)
    
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

    train_dataset = FaceSynthetics(data_root=data_root, 
                                    images=train_images,
                                    labels=train_labels,
                                    gt_labels=train_gt_labels,
                                    model_type=model_type,
                                    transform='train')
    val_dataset = FaceSynthetics(data_root=data_root, 
                                    images=val_images,
                                    labels=val_labels,
                                    gt_labels=val_gt_labels,
                                    model_type=model_type,
                                    transform='val')
    return train_dataset, val_dataset

def get_test_dataset(data_path:str, annot_path:str, model_type:str):
    images, labels, gt_labels = process_annot(annot_path, model_type)
    return FaceSynthetics(data_path, images, labels, gt_labels, "test")

class Heatmap_converter(object):
    def __init__(self, heatmap_size=96, window_size=7, sigma=1.75):
        self.heatmap_size = heatmap_size
        self.window_size = window_size
        self.pad_w = window_size // 2
        self.sigma = sigma
        self._generate_gaussian_kernel()
    def _generate_gaussian_kernel(self):
        gaussian_fun = lambda y, x : math.exp(-1 * (((self.pad_w - x) ** 2 + (self.pad_w - y) ** 2) / (2 * self.sigma * self.sigma)))
        self.kernel = torch.zeros((self.window_size, self.window_size))
        for y in range(self.window_size):
            for x in range(self.window_size):
                self.kernel[y, x] = gaussian_fun(y, x)

    def convert(self, landmark:torch.Tensor) -> torch.Tensor:
        """Convert landmark to heatmark
        Args:
            landmark: a torch tensor has shape(68,2)
        """
        heatmap = torch.zeros((landmark.shape[0], self.heatmap_size, self.heatmap_size)).float()
        for i, (x, y) in enumerate(landmark):
            
            ul_h  = (max(y - self.pad_w, 0), max(x - self.pad_w, 0)) # upper left point on heatmap
            lr_h = (min(y + self.pad_w + 1, self.heatmap_size), min(x + self.pad_w + 1, self.heatmap_size)) # lower right point on heatmap

            ul_k = (0 if (y - self.pad_w) > 0 else -1 * (y - self.pad_w), # upper left point on kernel
                    0 if (x - self.pad_w) > 0 else -1 * (x - self.pad_w)) 

 
            lr_k = ((self.window_size - max(-1 * ((self.heatmap_size - 1) - (y + self.pad_w)), 0)), # lower right point on kernel
                    (self.window_size - max(-1 * ((self.heatmap_size - 1) - (x + self.pad_w)), 0)))
            heatmap[i, ul_h[0]:lr_h[0], ul_h[1]:lr_h[1]] = self.kernel[ul_k[0]:lr_k[0], ul_k[1]:lr_k[1]]

        return heatmap

class FaceSynthetics(Dataset):
    def __init__(self, data_root:str, images:list, labels:np.ndarray, gt_labels:np.ndarray, transform="train", model_type="classifier", heatmap_size=96) -> None:
        """
        Args:
            data_root: the path of the data
            images: the path of the images
            labels: training labels
            gt_labels: groud truth labels
            transform: 
            model_type: the type of model, there are two option: "regrssion" or "classifier"
            heatmap_size: when model type == "classifier", then use this argument for generate heatmap
        """
        super(FaceSynthetics, self).__init__()
        self.data_root = data_root
        self.model_type = model_type
        # transform
        self.transform = get_transform(model_type=self.model_type,
                                        transform=transform)
        # data
        self.images = images
        self.labels= torch.tensor(labels)
        self.gt_labels = torch.tensor(gt_labels)
        self.num_classes = len(self.labels[0])
        # heatmap converter
        if self.model_type == "classifier":
            self.converter = Heatmap_converter(heatmap_size)
            self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int):
        # Read imagee
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im, label, gt_label = self.transform(im, self.labels[idx], self.gt_labels[idx])
        # transform point to heatmap
        if self.model_type == "classifier":
            label = self.converter.convert(label)
        return im, label, gt_label