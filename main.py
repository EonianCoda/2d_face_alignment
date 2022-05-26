import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.model import FAN
from utils.dataset import get_train_val_dataset
from utils.tool import fixed_seed, train
from cfg import cfg

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--use_image_ratio', type=float, default=1.0)
    # parser.add_argument('--batch_size', type=int, default='32')
    args = parser.parse_args()
    """ input argumnet """
    data_root = cfg['data_root']
    train_annot = cfg['train_annot']
    seed = cfg['seed']
    """ training hyperparameter """
    batch_size =  cfg['batch_size']
    lr = cfg['lr']
    split_ratio = cfg['split_ratio']
    epoch = cfg['epoch']
    # Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_seed(seed)

    model = FAN(num_HG=args.num_HG)
    print("Read annotation...")
    train_set, val_set = get_train_val_dataset(os.path.join(data_root, 'train') , train_annot, train_size=split_ratio, use_image_ratio=args.use_image_ratio)
    print("End of reading annotation!!!")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)

    scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    train(model, train_loader, val_loader, epoch, "./save", device, criterion, scheduler, optimizer)

    


if __name__ == '__main__':
    main()
