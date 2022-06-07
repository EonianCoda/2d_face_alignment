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

from scipy import interpolate
import matplotlib.pyplot as plt
import cv2

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


class AddBoundary(object):
    def __init__(self, num_landmarks=68):
        self.num_landmarks = num_landmarks

    def __call__(self, sample):
        landmarks_64 = np.floor(sample['label'].numpy() / 4.0)

        boundaries = {}
        boundaries['cheek'] = landmarks_64[0:17]
        boundaries['left_eyebrow'] = landmarks_64[17:22]
        boundaries['right_eyebrow'] = landmarks_64[22:27]
        boundaries['uper_left_eyelid'] = landmarks_64[36:40]
        boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [36, 41, 40, 39]])
        boundaries['upper_right_eyelid'] = landmarks_64[42:46]
        boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [42, 47, 46, 45]])
        boundaries['noise'] = landmarks_64[27:31]
        boundaries['noise_bot'] = landmarks_64[31:36]
        boundaries['upper_outer_lip'] = landmarks_64[48:55]
        boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [60, 61, 62, 63, 64]])
        boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [48, 59, 58, 57, 56, 55, 54]])
        boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
        functions = {}
        for key, points in boundaries.items():
            temp = points[0]
            new_points = points[0:1, :]
            for point in points[1:]:
                if point[0] == temp[0] and point[1] == temp[1]:
                    continue
                else:
                    new_points = np.concatenate((new_points, np.expand_dims(point, 0)), axis=0)
                    temp = point
            points = new_points
            if points.shape[0] == 1:
                points = np.concatenate((points, points+0.001), axis=0)
            k = min(4, points.shape[0])
            functions[key] = interpolate.splprep([points[:, 0], points[:, 1]], k=k-1,s=0)

        boundary_map = np.zeros((96, 96))

        fig = plt.figure(figsize=[96/96.0, 96/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(boundary_map, interpolation='nearest', cmap='gray')
        #ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, marker=',', c='w')

        for key in functions.keys():
            xnew = np.arange(0, 1, 0.01)
            out = interpolate.splev(xnew, functions[key][0], der=0)
            plt.plot(out[0], out[1], ',', linewidth=1, color='w')

        plt.savefig("test.png")

        img = fig2data(fig)
        
        plt.close()

        sigma = 1
        temp = 255-img[:,:,1]
        temp = cv2.distanceTransform(temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        temp = temp.astype(np.float32)
        temp = np.where(temp < 3*sigma, np.exp(-(temp*temp)/(2*sigma*sigma)), 0 )

        fig = plt.figure(figsize=[96/96.0, 96/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')
        ax.imshow(temp, cmap='gray')
        plt.close()

        boundary_map = fig2data(fig)

        sample['boundary'] = torch.from_numpy(boundary_map[:, :, 0]).float().div(255.0)

        return sample

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
                aug_setting:dict=None, heatmap_size=96, return_gt=True, 
                use_weight_map=False, fix_coord=False, data_weight=None,
                add_boundary=False) -> None:
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
        try:
            import google.colab
            self.IN_COLAB = True
        except:
            self.IN_COLAB = False
        img_path = "./data/train_imgs.pkl"
        img_data = "./data/train_data.pkl"
        if self.IN_COLAB and os.path.isfile(img_path) and os.path.isfile(img_data):
            import pickle
            old_images = pickle.load(open(img_path,'rb'))
            # Test if same
            for a,b in zip(self.images, old_images):
                a = os.path.basename(a)
                if a != b:
                    self.IN_COLAB = False
                    break
            del old_images
            if self.IN_COLAB:
                self.img_data = pickle.load(open(img_data,'rb'))
                print("Success Loading cached data!")
        else:
            self.IN_COLAB = False
        # Boundary

        if add_boundary:
            self.add_boundary = AddBoundary()
        else:
            self.add_boundary = None
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
        if fix_coord:
            self.converter = Heatmap_converter(heatmap_size)
        else:
            self.converter = Old_heatmap_converter(heatmap_size)

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
        # For colab
        if not self.IN_COLAB:
            idx = self.index_mapping(idx)
            # Read imagee
            img_path = os.path.join(self.data_root, self.images[idx])
            im = Image.open(img_path)
        else:
            im = self.img_data[idx]

        sample = {'img':im, 'label':self.labels[idx]}
        
        sample = self.transform(sample)
        if self.return_gt:
            sample['gt_label'] = sample['label'].clone()

        if self.add_boundary != None:
            sample = self.add_boundary(sample)

        # transform point to heatmap
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
        # transform
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.transform =  transforms.Compose([transforms.ToTensor()])
                                                #transforms.Normalize(means, stds)])
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