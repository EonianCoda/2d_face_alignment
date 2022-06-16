import random
import os
import pickle
import numpy as np

from dataset.FaceSynthetics import FaceSynthetics
from dataset.FaceSynthetics import Predicting_FaceSynthetics

def process_annot(annot_path:str):
    """Reading the annotation file and processing their labels (e.g. discard wrong label)
    """
    images, labels = pickle.load(open(annot_path, 'rb'))
    if isinstance(labels, list):
        labels = np.array(labels)
    mask = (labels >= 0) & (labels < 384) # shape = (bs, 68, 2)
    valid_idxs = mask.all(axis=(-1, -2)).nonzero()[0]
    
    labels = labels[valid_idxs]
    images = [images[i] for i in valid_idxs]
    return images, labels

def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8, use_image_ratio=1.0,
                            aug_setting:dict=None, use_weight_map=False,fix_coord=False, bg_negative=False):
    """Get training set and valiating set
    Args:
        data_root: the data root for images
        annot_path: thh path of the annotation file
        train_size: the size ratio of train:val
        use_image_ratio: how many images to use in training and validation
    """
    images, labels = process_annot(annot_path)
    # Split train/val set
    idxs = [i for i in range(int(len(images) * use_image_ratio))]
    random.shuffle(idxs)

    # Training set
    train_idxs = idxs[: int(len(idxs)*train_size)]
    train_images = [images[i] for i in train_idxs]
    train_labels = labels[train_idxs]

    # Validation set
    val_idxs = idxs[int(len(idxs)*train_size): ]
    val_images = [images[i] for i in val_idxs]
    val_labels = labels[val_idxs]
    # Calculate Euler angle

    train_dataset = FaceSynthetics(data_root=data_root, 
                                    images=train_images,
                                    labels=train_labels,
                                    return_gt=False,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
                                    bg_negative=bg_negative,
                                    transform='train',
                                    aug_setting=aug_setting)
    val_dataset = FaceSynthetics(data_root=data_root, 
                                    images=val_images,
                                    labels=val_labels,
                                    return_gt= True,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
                                    bg_negative=bg_negative,
                                    transform='val')
    return train_dataset, val_dataset

def get_test_dataset(data_path:str, annot_path:str):
    images, labels = process_annot(annot_path)
    test_dataset = FaceSynthetics(data_root=data_path, 
                                    images=images,
                                    labels=labels,
                                    return_gt= True,
                                    transform='test')
    return test_dataset

def get_pred_dataset(data_path:str):
    images = os.listdir(data_path)
    test_dataset = Predicting_FaceSynthetics(data_root=data_path, images=images)
    return test_dataset


