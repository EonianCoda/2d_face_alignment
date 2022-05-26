import torch
from torchvision.transforms import transforms
import random
import numpy as np


class Transform(object):
    def __init__(self, is_train=True, flip_x=0.5, random_noise=0.5):
        self.flip_x = flip_x
        self.random_noise = random_noise
        self.is_train = is_train

        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(means, stds)

    def __call__(self, img, label, gt_label):
        img = transforms.ToTensor()(img)

        # Random flip
        if self.is_train and random.random() > self.flip_x:
            # if label == None or gt_label == None:
            #     raise ValueError("Label or groud thruth label shouldn't be None!")
            img = transforms.RandomHorizontalFlip(1.0)(img)
            label = torch.tensor([95, 95]) - label
            gt_label = torch.tensor([383, 383]) - gt_label
        # Random noise
        if self.is_train and random.random() > self.random_noise:
            noise_num = int(random.random() * 0.1 * (147456))
            for _ in range(noise_num):
                prob = random.random()
                pos_x = int(383 * random.random())
                pos_y = int(383 * random.random())
                if prob > 0.5:
                    img[:, pos_y, pos_x] = 0
                else:
                    img[:, pos_y, pos_x] = 255
        img = self.normalize(img)

        return img, label, gt_label


def get_transform(data_type="train"):
    if data_type == "train":
        transform = Transform(is_train=True)
    elif data_type == "test" or data_type == "val":
        transform = Transform(is_train=False)
    return transform