# Torch
from copyreg import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.weighted_L2 import Weighted_L2
# Model
from model.tool import get_model
from dataset.tool import get_train_val_dataset, get_test_dataset, get_train_val_dataset_balanced
from utils.tool import fixed_seed, load_parameters, train
from utils.scheduler import Warmup_ReduceLROnPlateau
from losses.wing_loss import Adaptive_Wing_Loss, Wing_Loss
from cfg import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_image_ratio', type=float, default=1.0)
    parser.add_argument('--exp_name', help="the name of the experiment", default="", type=str)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--resume_model_path', help="the path of model weight for resuming training", type=str, default="")
    args = parser.parse_args()

    ### data setting ###
    train_data_root = cfg['train_data_root']
    train_annot = cfg['train_annot']
    test_data_root = cfg['test_data_root']
    test_annot = cfg['test_annot']
    use_image_ratio = args.use_image_ratio

    ## Data setting ###
    batch_size =  cfg['batch_size']
    split_ratio = cfg['split_ratio']
    balance_data = cfg['balance_data']
    aug_setting = cfg['aug_setting']
    add_boundary = cfg['add_boundary']
    ### training hyperparameter ###
    epoch = cfg['epoch']
    seed = cfg['seed']
    lr = cfg['lr']

    scheduler_type = cfg['scheduler_type']
    optimizer_type = cfg['optimizers'][cfg['optimizer_idx']]
    loss_type = cfg['losses'][cfg['loss_idx']]
    use_weight_map = (loss_type == "weighted_L2") or (loss_type == "adaptive_wing_loss")
    fix_coord = cfg['fix_coord']
    ### Resume ###
    resume = args.resume
    resume_epoch = args.resume_epoch
    resume_model_path = args.resume_model_path
    ### Logging parameter ###
    exp_name = args.exp_name # directory name of the tensorboard

    # Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_seed(seed)

    # Create model
    model = get_model(cfg)
    # Create train/val set
    print("Loading annotation...")
    if balance_data:
        print("Balance data !!! ")
        train_set, val_set = get_train_val_dataset_balanced(data_root=train_data_root, 
                                                annot_path=train_annot, 
                                                train_size=split_ratio,
                                                use_image_ratio=use_image_ratio,
                                                use_weight_map=use_weight_map,
                                                fix_coord=fix_coord,
                                                add_boundary=add_boundary,
                                                aug_setting=aug_setting)
        
    else:
        train_set, val_set = get_train_val_dataset(data_root=train_data_root, 
                                                annot_path=train_annot, 
                                                train_size=split_ratio,
                                                use_image_ratio=use_image_ratio,
                                                use_weight_map=use_weight_map,
                                                fix_coord=fix_coord,
                                                add_boundary=add_boundary,
                                                aug_setting=aug_setting)
    print("End of Loading annotation!!!")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    # Optimizer
    if optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=0.9, 
                                        weight_decay=1e-6)       
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=1e-6,
                                    nesterov=True)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=lr)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=lr,
                                    weight_decay=1e-6)
    # loss_type
    if loss_type == "L2":
        criterion = nn.MSELoss(reduction="sum")
    elif loss_type == "wing_loss":
        criterion = Wing_Loss()
    elif loss_type == "adaptive_wing_loss":
        criterion = Adaptive_Wing_Loss()
    elif loss_type == "weighted_L2":
        criterion = Weighted_L2()
    
    # Scheduler
    if scheduler_type == 0:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif scheduler_type == 1:
        warm_step = cfg['warm_step']
        patience = cfg['patience']
        scheduler = Warmup_ReduceLROnPlateau(optimizer, warm_step, patience=patience, verbose=True)
    model = model.to(device)
    
    # Testing data
    test_set = get_test_dataset(test_data_root, test_annot)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)


    # model save path
    save_path = f"./save/"

    # Resuming Training
    if resume:
        if resume_model_path == "":
            raise ValueError("If resume == True, then resume model path cannot be empty")
        load_parameters(model, resume_model_path, optimizer, scheduler)


    # Print training information
    print("Start training!!\n")
    print(f"Loss type = {loss_type}")
    print(f"Optimizer type = {optimizer_type}")
    print(f"Length of training dataloader = {len(train_loader)}")
    print(f"Resdiual Block = {cfg['resBlocks'][cfg['resBlock_idx']]}")
    print(f"Attention Block = {cfg['attention_blocks'][cfg['attention_block_idx']]}")
    print(f"Fix coord = {fix_coord}")
    print(f"Use CoordConv = {cfg['use_CoordConv']}")
    print(f"With_r = {cfg['with_r']}")
    print(f"Add CoordConv inHG = {cfg['add_CoordConv_inHG']}")
    print(f"Add add_boundary = {cfg['add_boundary']}")
    print("Aug setting = ", aug_setting)
    print(f"Balance_data = {balance_data}")
    
    aug = [k for k, v in cfg['aug_setting'].items() if v]
    aug = " ".join(aug)
    loss_type = cfg['losses'][cfg['loss_idx']]
    train_hyp = {'loss_type': loss_type,
                'optimizer': cfg['optimizers'][cfg['optimizer_idx']],
                'use_weight_map': (loss_type == "weighted_L2") or (loss_type == "adaptive_wing_loss"),
                'lr': cfg['lr'], 
                'use_image_ratio': use_image_ratio,
                'fix_coord': cfg['fix_coord'],
                'balance_data': cfg['balance_data'],
                'warm_step': cfg['warm_step'],
                'augmentation': aug,
                'seed': cfg['seed'],
                'add_boundary': cfg['add_boundary'],
                # model architecture
                'num_HG': cfg['num_HG'],
                'HG_depth': cfg['HG_depth'],
                'num_feats': cfg['num_feats'],
                'use_CoordConv':cfg['use_CoordConv'],
                'with_r': cfg['with_r'],
                'add_CoordConv_inHG':cfg['add_CoordConv_inHG'],
                'attention_block' : cfg['attention_blocks'][cfg['attention_block_idx']],
                'resBlock' : cfg['resBlocks'][cfg['resBlock_idx']]}

    

    train(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epoch=epoch,
        save_path=save_path,
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        loss_type=loss_type,
        exp_name=exp_name,
        fix_coord=fix_coord,
        add_boundary=add_boundary,
        train_hyp=train_hyp,
        resume_epoch=resume_epoch)

    


if __name__ == '__main__':
    main()
