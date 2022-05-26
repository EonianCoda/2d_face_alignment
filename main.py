import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.model import FAN
from utils.dataset import get_train_val_dataset
from utils.tool import Warmup_ReduceLROnPlateau, fixed_seed, train
from cfg import cfg

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--use_image_ratio', type=float, default=1.0)
    parser.add_argument('--exp_name', help="the name of the experiment", default="", type=str)
    parser.add_argument('--only_save_best', help="save all .pt or just save best result", action="store_true")
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
    scheduler_type = cfg['scheduler_type']
    """ Logging parameter """
    only_save_best = args.only_save_best
    exp_name = args.exp_name
    # Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_seed(seed)

    model = FAN(num_HG=args.num_HG)
    print("Loading annotation...")
    train_set, val_set = get_train_val_dataset(os.path.join(data_root, 'train') , train_annot, train_size=split_ratio, use_image_ratio=args.use_image_ratio)
    print("End of Loading annotation!!!")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    # Scheduler
    if scheduler_type == 0:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif scheduler_type == 1:
        warm_epoch = cfg['warm_epoch']
        scheduler = Warmup_ReduceLROnPlateau(optimizer, warm_epoch, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    train(model, train_loader, val_loader, epoch, "./save", device, criterion, scheduler, optimizer, exp_name, only_save_best)

    


if __name__ == '__main__':
    main()
