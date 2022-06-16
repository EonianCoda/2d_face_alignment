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


class Heatmap_converter(object):
    def __init__(self, heatmap_size=96, window_size=7, sigma=1.75, bg_negative=False):
        self.heatmap_size = heatmap_size
        self.window_size = window_size
        self.pad_w = window_size // 2
        self.sigma = sigma
        self.bg_negative = bg_negative
        self.bg_value = -0.05
        self.fg_ratio = 0.2 
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

        if self.bg_negative:
            heatmap = torch.ones((label.shape[0], self.heatmap_size, self.heatmap_size)).float() * self.bg_value
            max_coord, _ = label.max(dim=0)
            min_coord, _ = label.min(dim=0)
            # X coord
            max_x = int(max_coord[0])
            min_x = int(min_coord[0])
            offset_x = int(max_x * self.fg_ratio)
            max_x = min(max_x + offset_x, self.heatmap_size)
            min_x = max(min_x - offset_x, 0)
            # Y coord
            max_y = int(max_coord[1])
            min_y = int(min_coord[1])
            offset_y = int(max_y * self.fg_ratio)
            max_y = min(max_y + offset_y, self.heatmap_size)
            min_y = max(min_y - offset_y, 0)
            heatmap[:, min_y: max_y, min_x: max_x] = 0.0
        else:
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
    def __init__(self, data_root:str, images:list, labels:np.ndarray, transform="train", aug_setting:dict=None, heatmap_size=96, 
                return_gt=True, use_weight_map=False, fix_coord=False, bg_negative=False) -> None:
        """
        Args:
            data_root: the path of the data
            images: the path of the images
            labels: training labels
            gt_labels: groud truth labels
            transform: 
            heatmap_size: when model type == "classifier", then use this argument for generate heatmap
        """
        super(FaceSynthetics, self).__init__()
        self.data_root = data_root
        self.return_gt = return_gt
        self.use_weight_map = use_weight_map
        # transform
        self.transform = get_transform(data_type=transform,
                                        aug_setting=aug_setting)
        # data
        self.images = images
        self.labels= torch.tensor(labels)
        self.num_classes = len(self.labels[0])
        # For colab
        # try:
        #     import google.colab
        #     self.IN_COLAB = True
        #     img_path = "./data/train_imgs.pkl"
        #     img_data = "./data/train_data.pkl"
        #     if self.IN_COLAB and os.path.isfile(img_path) and os.path.isfile(img_data):
        #         import pickle
        #         old_images = pickle.load(open(img_path,'rb'))
        #         # Test if same
        #         for a,b in zip(self.images, old_images):
        #             a = os.path.basename(a)
        #             if a != b:
        #                 self.IN_COLAB = False
        #                 break
        #         del old_images
        #         if self.IN_COLAB:
        #             self.img_data = pickle.load(open(img_data,'rb'))
        #             print("Success Loading cached data!")
        #     else:
        #         self.IN_COLAB = False
        # except:
        #     self.IN_COLAB = False

        # heatmap converter
        if fix_coord:
            self.converter = Heatmap_converter(heatmap_size, bg_negative=bg_negative)
        else:
            self.converter = Old_heatmap_converter(heatmap_size)

    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def _generate_weight_map(heatmap):
        weight_map = torch.zeros_like(heatmap)
        k_size = 3
        for i in range(heatmap.shape[0]):
            dilate = ndimage.grey_dilation(heatmap[i], size=(k_size,k_size))
            weight_map[i][dilate>0.2] = 1
        return weight_map
    def __getitem__(self, idx:int):
        # For colab
        # if not self.IN_COLAB:
        #     # Read imagee
        #     img_path = os.path.join(self.data_root, self.images[idx])
        #     im = Image.open(img_path)
        # else:
        im = self.img_data[idx]

        sample = {'img':im, 'label':self.labels[idx]}
        sample = self.transform(sample)
        if self.return_gt:
            sample['gt_label'] = sample['label'].clone()

        # transform landmark to heatmap
        sample['label'] = self.converter.convert(sample['label'])

        if self.use_weight_map:
            sample['weight_map'] = self._generate_weight_map(sample['label'])
        return sample

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
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int):
        # Read imagee
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im = transforms.ToTensor()(im)
        return im