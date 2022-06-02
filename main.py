# Torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.weighted_L2 import Weighted_L2
# Model
from model.tool import get_model
from utils.dataset import get_train_val_dataset, get_test_dataset
from utils.tool import fixed_seed, load_parameters, train
from utils.scheduler import Warmup_ReduceLROnPlateau
from losses.wing_loss import Adaptive_Wing_Loss, Wing_Loss
from cfg import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--use_image_ratio', type=float, default=1.0)
    parser.add_argument('--exp_name', help="the name of the experiment", default="", type=str)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--resume_model_path', help="the path of model weight for resuming training", type=str, default="")
    parser.add_argument('--only_save_best', help="save all .pt or just save best result", action="store_true")
    args = parser.parse_args()

    ### data setting ###
    train_data_root = cfg['train_data_root']
    train_annot = cfg['train_annot']
    use_image_ratio = args.use_image_ratio
    test_data_root = cfg['test_data_root']
    test_annot = cfg['test_annot']

    ### model setting ###
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        cfg.update(classifier_cfg)
        num_HG = cfg['num_HG']
        HG_depth = cfg['HG_depth']
        num_feats = cfg['num_feats']
    elif model_type == "regressor":
        cfg.update(regressor_cfg)
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']

    ### training hyperparameter ###
    scheduler_type = cfg['scheduler_type']
    batch_size =  cfg['batch_size']
    split_ratio = cfg['split_ratio']
    epoch = cfg['epoch']
    seed = cfg['seed']
    lr = cfg['lr']
    loss_type = cfg['losses'][cfg['loss_idx']]
    aug_setting = cfg['aug_setting']
    fix_coord = cfg['fix_coord']
    ### Resume ###
    resume = args.resume
    resume_epoch = args.resume_epoch
    resume_model_path = args.resume_model_path
    ### Logging parameter ###
    only_save_best = args.only_save_best
    exp_name = args.exp_name # directory name of the tensorboard

    # Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_seed(seed)

    # Create model
    model = get_model(cfg)
    # Create train/val set
    print("Loading annotation...")
    if loss_type == "weighted_L2":
        use_weight_map = True
    else:
        use_weight_map = False
    train_set, val_set = get_train_val_dataset(data_root=train_data_root, 
                                               annot_path=train_annot, 
                                               train_size=split_ratio,
                                               use_image_ratio=use_image_ratio,
                                               model_type=model_type,
                                               use_weight_map=use_weight_map,
                                               fix_coord=fix_coord,
                                               aug_setting=aug_setting)
    print("End of Loading annotation!!!")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    # Optimizer adn criterion
    if model_type == "classifier":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=0.9, 
                                        weight_decay=1e-6)
    elif model_type == "regressor":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=0.9, 
                                        weight_decay=1e-6)
    # loss_type
    if loss_type == "L2":
        if model_type == "classifier":
            criterion = nn.MSELoss(reduction="sum")
        elif model_type == "regressor":
            criterion = nn.MSELoss()
    elif loss_type == "L1":
        if model_type == "classifier":
            raise ValueError("If model type == 'classifier', then loss_type cannot be L1")
        criterion = nn.L1Loss()
    elif loss_type == "smooth_L1":
        if model_type == "classifier":
            raise ValueError("If model type == 'classifier', then loss_type cannot be smooth_L1")
        criterion = nn.SmoothL1Loss()
    elif loss_type == "wing_loss":
        criterion = Wing_Loss()
    elif loss_type == "adaptive_wing_loss":
        if model_type == "regressor":
            raise ValueError("If model type == 'regressor', then loss_type cannot be Adaptive_Wing_Loss")
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
    test_set = get_test_dataset(test_data_root, test_annot, model_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)


    # model save path
    save_path = f"./save/{model_type}"

    if resume:
        if resume_model_path == "":
            raise ValueError("If resume == True, then resume model path cannot be empty")
        load_parameters(model, resume_model_path, optimizer, scheduler, model_type)


    aug = [k for k, v in aug_setting.items() if v]
    aug = " ".join(aug)
    train_hyp = {'lr':lr, 
                'bsize': batch_size,
                'model_type': model_type,
                'loss_type':loss_type,
                'use_image_ratio': use_image_ratio,
                'warm_step': warm_step,
                'augmentation':aug}

    if model_type == "classifier":
        train_hyp['num_HG'] = num_HG
        train_hyp['HG_depth'] = HG_depth
    elif model_type == "regressor":
        train_hyp['backbone'] = backbone
        train_hyp['dropout'] = dropout


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
        model_type=model_type,
        loss_type=loss_type,
        exp_name=exp_name,
        train_hyp=train_hyp,
        only_save_best=only_save_best,
        fix_coord=fix_coord,
        resume_epoch=resume_epoch)

    


if __name__ == '__main__':
    main()
