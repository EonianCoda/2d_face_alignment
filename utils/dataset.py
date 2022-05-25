import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from platform import python_version


def convert_label(origin_label:list):
    label = np.array(origin_label)
    if (label >= 384).any():
        return False, None
    label = np.round(label * 0.25)

    same_count = 0
    while True:
        flag = True
        idxs = np.lexsort((label[:,1], label[:,0]))
        for i in range(len(idxs) - 1):
            cur_idx, next_idx = idxs[i], idxs[i + 1]
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

    label = label.astype(np.int32)
    return True, label

def get_transform(data_type="train"):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    if data_type == "train":
        transform = transforms.Compose([#transforms.RandomPerspective(distortion_scale=0.3),
                                        transforms.ToTensor(),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.Normalize(means, stds),
                                    ])
    elif data_type == "test" or data_type == "val":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(means, stds),
                                    ])
    return transform

def get_train_val_dataset(data_root:str, annot_path:str, train_size=0.8):

    images, labels = process_annot(annot_path)

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=train_size)
    train_dataset = FaceSynthetics(data_root, train_images, train_labels, get_transform('train'))
    val_dataset = FaceSynthetics(data_root, val_images, val_labels, get_transform('val'))
    return train_dataset, val_dataset

def process_annot(annot_path:str):
    # If python verions < 3.8.0, then use pickle5
    py_version = python_version()
    py_version = int(''.join(py_version.split('.')))
    if py_version < 380:
        import pickle5 as pickle
    else:
        import pickle
    
    images, labels = pickle.load(open(annot_path, 'rb'))
    
    valid_imgs = []
    valid_labels = []
    for img, label in zip(images, labels):
        result = convert_label(label)
        if result[0]:
            valid_imgs.append(img)
            valid_labels.append(result[1])
    return valid_imgs, valid_labels

class FaceSynthetics(Dataset):
    def __init__(self, data_root:str, images:list, labels:list, transform=None, heatmap_size=96) -> None:
        super(FaceSynthetics, self).__init__()
        self.data_root = data_root
        self.images = images
        self.labels= labels
        self.transform = transform 
        
        self.heatmap_size = heatmap_size

    def gen_heatmap(self, label):
        heatmap = np.zeros((68, self.heatmap_size, self.heatmap_size))
        for i, (x, y) in enumerate(label):
            heatmap[i, y, x] = 1
        return torch.tensor(heatmap).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int):
        img_path = os.path.join(self.data_root, self.images[idx])
        im = Image.open(img_path)
        im = self.transform(im)
        label = self.gen_heatmap(self.labels[idx])
        return im, label