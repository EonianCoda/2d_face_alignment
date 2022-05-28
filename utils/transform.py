import torch
from torchvision.transforms import transforms
import random

class RandomHorizontalFlip(object):
    def __init__(self, flip_x=0.5):
        self.flip_x = flip_x
        self.mapping = [[0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 41, 40, 31, 32, 50, 49, 48, 61, 60, 67, 59, 58], 
                        [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 45, 44, 43, 42, 46, 47, 35, 34, 52, 53, 54, 63, 64, 65, 55, 56]]
        self.do_mapping = True
    def __call__(self, img, label:torch.Tensor, gt_label:torch.Tensor):
        """
        Args:
            img: the PIL image
        """
        h, w = img.size
        max_size_on_label = int(h / 4)
        max_size = h
        if random.random() < self.flip_x:
            img = transforms.RandomHorizontalFlip(1.0)(img)

            # Flip x coordinate
            label[:, 0] = (max_size_on_label - 1) - label[:, 0]
            gt_label[:, 0] = (max_size - 1) - gt_label[:, 0]
            if self.do_mapping:
                tmp = label[self.mapping[1], ...].clone()
                label[self.mapping[1], ...] = label[self.mapping[0], ...]
                label[self.mapping[0], ...] = tmp

                tmp = gt_label[self.mapping[1], ...].clone()
                gt_label[self.mapping[1], ...] = gt_label[self.mapping[0], ...]
                gt_label[self.mapping[0], ...] = tmp
        return img, label, gt_label

class RandomNoise(object):
    def __init__(self, prob=0.5, ratio=0.05):
        self.prob = prob
        self.ratio = ratio
    def __call__(self, img):
        c, h, w = img.shape
        if random.random() < self.prob:
            noise_num = int(random.random() * self.ratio * h * w)
            for _ in range(noise_num):
                prob = random.random()
                pos_x = int((w - 1) * random.random())
                pos_y = int((h - 1) * random.random())
                if prob > 0.5:
                    img[:, pos_y, pos_x] = 0.0
                else:
                    img[:, pos_y, pos_x] = 1.0 
        return img

class Transform(object):
    def __init__(self, is_train=True, flip_x=0.5, random_noise=0.5, noise_ratio=0.1):
        self.flip_x = flip_x
        self.random_noise = random_noise
        self.is_train = is_train

        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(means, stds)
        self.random_flip = RandomHorizontalFlip(flip_x)
        self.random_noise = RandomNoise(random_noise, noise_ratio)
    def __call__(self, img, label, gt_label):
        
        # Random flip
        if self.is_train:
            img, label, gt_label = self.random_flip(img, label, gt_label)

        img = transforms.ToTensor()(img)

        # # Random noise
        # if self.is_train:
        #     img = self.random_noise(img)

        img = self.normalize(img)
        return img, label, gt_label


def get_transform(data_type="train"):
    if data_type == "train":
        transform = Transform(is_train=True)
    elif data_type == "test" or data_type == "val":
        transform = Transform(is_train=False)
    return transform