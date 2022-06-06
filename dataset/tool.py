import torch
from platform import python_version
import random
import os
import numpy as np
import math
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

from dataset.FaceSynthetics import FaceSynthetics
from dataset.FaceSynthetics import Predicting_FaceSynthetics

class PDB(object):
    """Pose-based data balancing
    """
    def __init__(self, annot_path:str):
        path = os.path.dirname(annot_path)
        file_name = os.path.basename(annot_path).split('.')[0]
        cached_file = f'cached_{file_name}_projected.pkl'
        self.cached_file = os.path.join(path, cached_file)

    def _cal_projected(self, labels):
        import pickle
        # Load cached file
        if os.path.isfile(self.cached_file):
            self.projected = pickle.load(open(self.cached_file, 'rb'))
            return

        if isinstance(labels, torch.Tensor):
            labels = labels.clone()
        elif isinstance(labels, np.ndarray):
            labels = labels.copy()

        print("Calculating projected....")
        shapes = labels
        ref_shape = shapes.mean(axis=0)
        aligned = []
        for shape in shapes:
            _ , transform , _ = procrustes(ref_shape, shape)
            aligned.append(transform)
        aligned = np.stack(aligned)

        b, n, c = aligned.shape # (batch_size, num_landmark, coordinate)
        pca = PCA(n_components=1)
        self.projected = pca.fit_transform(aligned.reshape((-1, n*c)))
        self.projected = self.projected[:,0]
        print("End of calculating projected....")

        pickle.dump(self.projected, open(self.cached_file, 'wb'))

    def get_weights(self, labels):
        # Calculating projected
        self._cal_projected(labels)
        num_data = len(self.projected)
        rank = np.argsort(self.projected)
        value_rank = np.sort(self.projected)

        bins = [-0.4, -0.3, 0.3, 0.4]
        indexs = [(value_rank<bin).sum() for bin in bins]
        indexs.append(num_data)

        W = [1, 1, 2, 3, 3]
        weights = np.zeros((num_data))
        
        cur_idx = 0
        for i, w in enumerate(W):
            target_index = rank[cur_idx: indexs[i]]
            weights[target_index] = w
            cur_idx = indexs[i]

        return weights


def get_python_version():
    py_version = python_version()
    py_version = int(''.join(py_version.split('.')[:2]))
    return py_version

def process_annot(annot_path:str):
    """Reading the annotation file and processing their labels (e.g. discard wrong label)
    """
    # If python verions < 3.8.0, then use pickle5
    if get_python_version() < 38:
        import pickle5 as pickle
    else:
        import pickle

    images, labels = pickle.load(open(annot_path, 'rb'))
    mask = (labels >= 0) & (labels < 384) # shape = (bs, 68, 2)
    valid_idxs = mask.all(axis=(-1, -2)).nonzero()[0]
    
    labels = labels[valid_idxs]
    images = [images[i] for i in valid_idxs]
    return images, labels

def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8, use_image_ratio=1.0,
                            aug_setting:dict=None, use_weight_map=False,fix_coord=False, get_weight=False):
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

    if get_weight:
        pdb = PDB(annot_path)
        weights = pdb.get_weights(labels)
        train_weights = weights[train_idxs]
    else:
        train_weights = None

    train_dataset = FaceSynthetics(data_root=data_root, 
                                    images=train_images,
                                    labels=train_labels,
                                    return_gt=False,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
                                    data_weight = train_weights,
                                    transform='train',
                                    aug_setting=aug_setting)
    val_dataset = FaceSynthetics(data_root=data_root, 
                                    images=val_images,
                                    labels=val_labels,
                                    return_gt= True,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
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


