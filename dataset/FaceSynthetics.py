import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import math

from dataset.transform import get_transform
from utils.convert_tool import is_None


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

    def _get_kernel(self, landmark_i:int, offset:torch.Tensor):
        """Get 7X7 kernel from origin 9X9 kernel with offset
        Args:
            offset: a torch tensor has shape(2,)
        """
        def convert(x):
            if x > 0:
                return float(1 - x), float(x)
            else:
                return float(-x), float(1 + x)
        x, y = (offset / 2) * 0.5
        l_x, r_x = convert(x)
        t_y, b_y = convert(y)

        # Set weight kernel's weight
        self.weight_kernel.weight.data = torch.tensor([[[[r_x * b_y, l_x * b_y],
                                                        [r_x * t_y, l_x * t_y]]]])

        start_x = 1 if x < 0 else 0 
        start_y = 1 if y < 0 else 0
        kernel = self.kernel[start_y: start_y + self.window_size +1, start_x: start_x + self.window_size +1]

        kernel = kernel.unsqueeze(dim=0)
        kernel = self.weight_kernel(kernel).squeeze(dim=0)
        # Normalize kernel

        middle_p = float(kernel[self.window_size // 2, self.window_size // 2])

        kernel /= middle_p

        return kernel
    def convert(self, label:torch.Tensor) -> torch.Tensor:
        """Convert landmark to heatmap
        Args:
            label: a torch tensor has shape(68,2)
        """
        gt_label = label.clone()
        label = torch.round(label.clone() * 0.25).long()
        offsets = (gt_label - label *4)

        heatmap = torch.zeros((label.shape[0], self.heatmap_size, self.heatmap_size)).float()
        for i, (offset, (x, y)) in enumerate(zip(offsets, label)):
            kernel = self._get_kernel(i, offset)

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
            num_normal_data = (data_weight == 2).sum()
            print(num_normal_data, num_normal_data / len(self.images))

            idxs = np.arange(len(self.images))

            mapping_idxs = [idxs[data_weight == 2]]
            # Type1
            mask = (data_weight == 1)
            mapping_idxs.append(np.random.choice(idxs[mask], max(int(num_normal_data * 0.5), mask.sum())))
            # Type3
            mask = (data_weight == 3)
            mapping_idxs.append(np.random.choice(idxs[mask], max(int(num_normal_data * 0.5), mask.sum())))

            mapping_idxs = np.concatenate(mapping_idxs).flatten()
            np.random.shuffle(mapping_idxs)
            self.mapping_idxs = mapping_idxs
            self.num_data = len(self.mapping_idxs)
        else:
            self.num_data = len(self.images)
            self.mapping_idxs = [i for i in range(self.num_data)]


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
        return self.mapping_idxs[idx]
        # if self.extra_num_data == 0:
        #     return idx
        # else:
        #     if idx >= self.max_idx:
        #         idx = np.random.choice(range(self.max_idx), p=self.data_weight)
        #         return int(idx)
        #     else:
        #         return idx
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