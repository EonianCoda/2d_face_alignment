import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy import ndimage

from PIL import Image
import numpy as np
from platform import python_version
import os
import random
import math
import copy

from utils.transform import get_transform
from utils.convert_tool import is_None

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
    mask = (labels >= 0) & (labels < 384) # shape = (bs, 68, 2)
    valid_idxs = mask.all(axis=(-1, -2)).nonzero()[0]
    
    labels = labels[valid_idxs]
    images = [images[i] for i in valid_idxs]
    return images, labels

    # For classifier
    # Generate kernel
    # path = os.path.dirname(annot_path)
    # file_name = os.path.basename(annot_path).split('.')[0]
    # cached_file = f'cached_{file_name}_kernel.pkl'
    # cached_file = os.path.join(path, cached_file)

    # if os.path.isfile(cached_file):
    #     kernels = pickle.load(open(cached_file, 'rb'))
    #     return images, labels, kernels
    # else:
    #     converter = Heatmap_converter()
    #     labels = torch.from_numpy(labels)
    #     gt_labels = labels.copy()
    #     labels = torch.round(labels * 0.25).long()
    #     offsets = (gt_labels - labels *4)
    #     kernels = []
    #     for offset in offsets:
    #         kernels.append(converter._get_kernel(offset))
    #     with open(cached_file, 'wb') as f:
    #         pickle.dump(kernels, f)


    #     return images, gt_labels, kernels

def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8, use_image_ratio=1.0, model_type="classifier",
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
                                    model_type=model_type,
                                    return_gt=False,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
                                    data_weight = train_weights,
                                    transform='train',
                                    aug_setting=aug_setting)
    val_dataset = FaceSynthetics(data_root=data_root, 
                                    images=val_images,
                                    labels=val_labels,
                                    model_type=model_type,
                                    return_gt= True,
                                    use_weight_map=use_weight_map,
                                    fix_coord=fix_coord,
                                    transform='val')
    return train_dataset, val_dataset

def get_test_dataset(data_path:str, annot_path:str, model_type:str):
    images, labels = process_annot(annot_path)
    test_dataset = FaceSynthetics(data_root=data_path, 
                                    images=images,
                                    labels=labels,
                                    model_type=model_type,
                                    return_gt= True,
                                    transform='test')
    return test_dataset

def get_pred_dataset(data_path:str):
    images = os.listdir(data_path)
    test_dataset = Predicting_FaceSynthetics(data_root=data_path, images=images)
    return test_dataset

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
        # num_data = [indexs[0],
        #             indexs[1] - indexs[0], 
        #             indexs[2] - indexs[1],
        #             indexs[3] - indexs[2],
        #             len(projected) - indexs[3]]

        # ratio = [num / len(projected) for num in num_data]
        # weight = 1 / (ratio / ratio[2])
        W = [3,2,1,2,3]
        weights = np.zeros((num_data))
        
        cur_idx = 0
        for i, w in enumerate(W):
            target_index = rank[cur_idx: indexs[i]]
            weights[target_index] = w
            cur_idx = indexs[i]

        return weights

class Heatmap_converter(object):
    def __init__(self, heatmap_size=96, window_size=7, sigma=1.75):
        self.heatmap_size = heatmap_size
        self.window_size = window_size
        self.pad_w = window_size // 2
        self.sigma = sigma
        self._generate_gaussian_kernel()

    def _generate_gaussian_kernel(self):
        # Kernel
        kernel_w = self.window_size + 2
        kernel_pad_w = kernel_w // 2
        gaussian_fun = lambda y, x : math.exp(-1 * (((kernel_pad_w - x) ** 2 + (kernel_pad_w - y) ** 2) / (2 * self.sigma * self.sigma)))
        self.kernel = torch.zeros((kernel_w, kernel_w))
        for y in range(kernel_w):
            for x in range(kernel_w):
                self.kernel[y, x] = gaussian_fun(y, x)

        self.kernel = self.kernel.float()
        self.weight_kernel = nn.Conv2d(1,1,kernel_size=2 ,stride=1, bias=False)
        self.weight_kernel.weight.requires_grad = False

    def _get_kernel(self, offset:torch.Tensor):
        def convert(x):
            if x > 0:
                return (1 - x), x
            else:
                return -x, 1 + x
        x, y = (offset / 2) * 0.5
        l_x, r_x = convert(x)
        t_y, b_y = convert(y)
        weight = torch.tensor([[[[r_x * b_y, l_x * b_y],
                                [r_x * t_y, l_x * t_y]]]])

        start_x = 1 if x < 0 else 0 
        start_y = 1 if y < 0 else 0

        kernel = self.kernel[start_y: start_y + self.window_size +1, start_x: start_x + self.window_size +1]
        kernel = kernel.unsqueeze(dim=0)
        self.weight_kernel.weight.data = weight

        kernel = self.weight_kernel(kernel).squeeze(dim=0)
        kernel /= kernel[self.window_size // 2, self.window_size // 2].clone()
        return kernel
    def convert(self, label:torch.Tensor) -> torch.Tensor:
        """Convert landmark to heatmark
        Args:
            label: a torch tensor has shape(68,2)
        """
        gt_label = label.clone()
        label = torch.round(label.clone() * 0.25).long()
        offsets = (gt_label - label *4)

        heatmap = torch.zeros((label.shape[0], self.heatmap_size, self.heatmap_size)).float()
        for i, (offset, (x, y)) in enumerate(zip(offsets, label)):
            kernel = self._get_kernel(offset)

            ul_h  = (max(y - self.pad_w, 0), max(x - self.pad_w, 0)) # upper left point on heatmap
            lr_h = (min(y + self.pad_w + 1, self.heatmap_size), min(x + self.pad_w + 1, self.heatmap_size)) # lower right point on heatmap

            ul_k = (0 if (y - self.pad_w) > 0 else -1 * (y - self.pad_w), # upper left point on kernel
                    0 if (x - self.pad_w) > 0 else -1 * (x - self.pad_w)) 

            lr_k = ((self.window_size - max(-1 * ((self.heatmap_size - 1) - (y + self.pad_w)), 0)), # lower right point on kernel
                    (self.window_size - max(-1 * ((self.heatmap_size - 1) - (x + self.pad_w)), 0)))
            heatmap[i, ul_h[0]:lr_h[0], ul_h[1]:lr_h[1]] = kernel[ul_k[0]:lr_k[0], ul_k[1]:lr_k[1]]

        return heatmap

class Old_heatmap_converter(object):
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
        landmark = torch.round(landmark.clone() * 0.25).long()
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
    def __init__(self, data_root:str, images:list, labels:np.ndarray, transform="train", 
                model_type="classifier", aug_setting:dict=None, heatmap_size=96, return_gt=True, 
                use_weight_map=False, fix_coord=False, data_weight=None) -> None:
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
        self.return_gt = return_gt
        self.use_weight_map = use_weight_map
        # transform
        self.transform = get_transform(data_type=transform,
                                        aug_setting=aug_setting)
        # data
        self.images = images
        self.labels= torch.tensor(labels)
        self.num_classes = len(self.labels[0])

        # data weight
        if not is_None(data_weight):
            max_ratio = 0.2
            num_imgs = len(self.images)
            total_weight = int(data_weight.sum())
            self.extra_num_data = min( int((data_weight != 1).sum() * max_ratio), int(num_imgs * max_ratio))
            self.num_data = num_imgs + self.extra_num_data
            self.max_idx = len(self.images)
            data_weight = data_weight - 1
            self.data_weight = data_weight / data_weight.sum()
        else:
            self.num_data = len(self.images)
            self.extra_num_data = 0

        # heatmap converter
        if self.model_type == "classifier":
            if fix_coord:
                self.converter = Heatmap_converter(heatmap_size)
            else:
                self.converter = Old_heatmap_converter(heatmap_size)
            # self.heatmap_size = heatmap_size

    def __len__(self):
        return self.num_data
    
    def index_mapping(self, idx:int):
        if self.extra_num_data == 0:
            return idx
        else:
            if idx >= self.max_idx:
                idx = np.random.choice(range(self.max_idx), p=self.data_weight)
                return int(idx)
            else:
                return idx
    @staticmethod
    def _generate_weight_map(heatmap):
        weight_map = torch.zeros_like(heatmap)
        k_size = 3
        for i in range(heatmap.shape[0]):
            dilate = ndimage.grey_dilation(heatmap[i], size=(k_size,k_size))
            weight_map[i][dilate>0.2] = 1
        return weight_map
    def __getitem__(self, idx:int):
        idx = self.index_mapping(idx)
        # Read imagee
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im, label = self.transform(im, self.labels[idx])
        if self.return_gt:
            gt_label = label.clone()
        # transform point to heatmap
        if self.model_type == "classifier":
            label = self.converter.convert(label)

        if self.return_gt and self.use_weight_map:
            return im, label, gt_label, self._generate_weight_map(label)
        elif self.return_gt:
            return im, label, gt_label
        elif self.use_weight_map:
            return im, label, self._generate_weight_map(label)
        else:
            return im, label

class Predicting_FaceSynthetics(Dataset):
    def __init__(self, data_root:str, images:list) -> None:
        """
        Args:
            data_root: the path of the data
            images: the path of the images
            labels: training labels
        """
        super(Predicting_FaceSynthetics, self).__init__()
        self.data_root = data_root
        # transform
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.transform =  transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(means, stds)])
        # data
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int):
        # Read imagee
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im = self.transform(im)
        return im