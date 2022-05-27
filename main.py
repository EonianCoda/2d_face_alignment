import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.model import FAN
from utils.transform import get_transform
from utils.dataset import FaceSynthetics, get_train_val_dataset, process_annot
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
    args = parser.parse_args()

    ### data setting ###
    train_data_root = cfg['train_data_root']
    train_annot = cfg['train_annot']
    use_image_ratio = args.use_image_ratio

    test_data_root = cfg['test_data_root']
    test_annot = cfg['test_annot']
    ### model setting ###
    num_HG = args.num_HG
    ### training hyperparameter ###
    scheduler_type = cfg['scheduler_type']
    batch_size =  cfg['batch_size']
    split_ratio = cfg['split_ratio']
    epoch = cfg['epoch']
    seed = cfg['seed']
    lr = cfg['lr']
    

    ### Logging parameter ###
    only_save_best = args.only_save_best
    exp_name = args.exp_name # directory name of the tensorboard
    # Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_seed(seed)

    # Create model
    model = FAN(num_HG=num_HG)

    # Create train/val set
    print("Loading annotation...")
    train_set, val_set = get_train_val_dataset(data_root=train_data_root, 
                                               annot_path=train_annot, 
                                               train_size=split_ratio, 
                                               use_image_ratio=use_image_ratio)
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
    
    # Testing data
    images, labels, gt_labels = process_annot(test_annot)
    test_set = FaceSynthetics(test_data_root, images, labels, gt_labels, get_transform("test"))
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    train(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epoch=epoch,
        save_path="./save",
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        exp_name=exp_name,
        only_save_best=only_save_best)

    


if __name__ == '__main__':
    main()
