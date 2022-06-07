from cProfile import label
import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import random
import math

class RandomPadding(object):
    def __init__(self, prob=0.5, padding:int=80):
        self.prob = prob
        self.padding = padding
        self.pad = F.pad
        self.resize = F.resize
    def __call__(self, sample:dict):
        """
        Args:
            img: the PIL image
        """
        h, w = sample['img'].size
        
        if random.random() < self.prob:
            pad_size = int(self.padding * min(random.random(), 0.5))
            new_h = h + pad_size * 2
            sample['img'] = self.pad(sample['img'], padding=pad_size, padding_mode="edge")
            sample['img'] = self.resize(sample['img'], (h, w))

            ratio = h / new_h
            sample['label'] *= ratio
            sample['label'] += pad_size * ratio
            
        return sample

class RandomRoation(object):
    def __init__(self, img_shape:tuple=(384,384,3), prob=0.5, angle=(-30, 30)):
        """Random roation
        Args:
            img_shape: the shape of input image, must be (h,w,c)
        """
        self.prob = prob
        self.angle = angle
        self.angle_list = [self.angle[0] + i for i in range(self.angle[1] - self.angle[0])]
        self.img_shape = img_shape
        self._generate_rotation_matrix()

    def _generate_rotation_matrix(self):
        self.rot_matrices = []
        for angle in self.angle_list:
            degree = angle / (180 / math.pi)
            cos , sin = math.cos(degree), math.sin(degree) 
            rot_matrix = torch.tensor([[cos, sin],[-sin, cos]])
            self.rot_matrices.append(rot_matrix.float())

    @staticmethod
    def _rotate_points(points, w, h, rot_matrix):
        center = torch.tensor([[w / 2, h / 2]])
        points -= center
        points = torch.matmul(rot_matrix, points.T)
        points = points.T + center
        return points

    def __call__(self, sample:dict):

        if random.random() < self.prob:
            label = sample['label'].clone()
            img = sample['img']

            h, w, c = self.img_shape
            
            angle_i = random.randint(0, len(self.angle_list)-1)
            angle = self.angle_list[angle_i]
            r_img = F.rotate(img,  angle)

            rot_matrix = self.rot_matrices[angle_i]

            r_label = self._rotate_points(label, w, h, rot_matrix) # Rotate label
            # Out of bound
            if (r_label < 0).any() or (r_label >= h).any():
                return sample

            sample['label'] = r_label
            sample['img'] = r_img
            return sample

        return sample

class RandomHorizontalFlip(object):
    def __init__(self, flip_x=0.5):
        self.flip_x = flip_x
        self.mapping = [[0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 41, 40, 31, 32, 50, 49, 48, 61, 60, 67, 59, 58], 
                        [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 45, 44, 43, 42, 46, 47, 35, 34, 52, 53, 54, 63, 64, 65, 55, 56]]
        self.do_mapping = True
    def __call__(self, sample:torch.Tensor):
        """
        Args:
            img: the PIL image
        """
        if random.random() < self.flip_x:
            label = sample['label']
            h, w = sample['img'].size

            sample['img'] = F.hflip(sample['img']) #transforms.RandomHorizontalFlip(1.0)(img)
            # Flip x coordinate
            label[:, 0] = (h - 1) - label[:, 0]
            if self.do_mapping:
                tmp = label[self.mapping[1], ...].clone()
                label[self.mapping[1], ...] = label[self.mapping[0], ...]
                label[self.mapping[0], ...] = tmp

        return sample

class RandomNoise(object):
    def __init__(self, prob=0.5, ratio=0.1):
        self.prob = prob
        self.ratio = ratio
    def __call__(self, sample):
        
        if random.random() < self.prob:
            c, h, w = sample['img'].shape
            noise_num = int(random.random() * self.ratio * h * w)
            for _ in range(noise_num):
                prob = random.random()
                pos_x = int((w - 1) * random.random())
                pos_y = int((h - 1) * random.random())
                if prob > 0.5:
                    sample['img'][:,pos_y, pos_x] = 0.0
                else:
                    sample['img'][:,pos_y, pos_x] = 1.0 
        return sample

class Transform(object):
    def __init__(self, is_train=True, aug_setting:dict=None):
        self.is_train = is_train
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(means, stds)
        
        if self.is_train:
            if aug_setting == None:
                raise ValueError("When is_train == 'True', aug setting cannot be None ")
            self.aug_setting = aug_setting
            self.random_flip = RandomHorizontalFlip()
            self.random_noise = RandomNoise()
            self.random_rotation = RandomRoation()
            self.gaussian_blur = transforms.GaussianBlur((7,7), sigma=(1.0, 2.0))
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            self.gray_transform = transforms.Grayscale(num_output_channels=3)
            self.random_padding = RandomPadding()

    def __call__(self, sample:dict):
        sample['label'] = sample['label'].clone()
        # Random Padding
        if self.is_train and self.aug_setting['padding']:
            sample = self.random_padding(sample) 

        # Random flip
        if self.is_train and self.aug_setting['flip']:
            sample = self.random_flip(sample)

        # Random rotation
        if self.is_train and self.aug_setting['rotation']:
            sample = self.random_rotation(sample)

        sample['img'] = transforms.ToTensor()(sample['img'])

        # Gaussian Blur
        if self.is_train and self.aug_setting['gaussianBlur']:
            if random.random() > 0.5:
                sample['img'] = self.gaussian_blur(sample['img'])
        
        # Color Jitter
        if self.is_train and self.aug_setting['colorJitter']:
            if random.random() > 0.5:
                sample['img'] = self.color_jitter(sample['img'])
        
        # Random noise
        if self.is_train and self.aug_setting['noise']:
            sample = self.random_noise(sample)

        if self.is_train and self.aug_setting['grayscale']:
            #prob = 2/3
            if random.random() > 2/3:
                sample['img'] = self.gray_transform(sample['img'])

        #img = self.normalize(img)
        return sample


def get_transform(data_type="train", aug_setting:dict=None):
    if data_type == "train":
        transform = Transform(is_train=True, aug_setting=aug_setting)
    elif data_type == "test" or data_type == "val":
        if aug_setting != None:
            raise ValueError('If data_type == "test" or "val", then aug_setting should be None')
        transform = Transform(is_train=False)
    return transform
