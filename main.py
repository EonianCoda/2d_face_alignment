import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.FAN import FAN
from model.Regression import RegressionModel
from utils.dataset import get_train_val_dataset, get_test_dataset
from utils.tool import Warmup_ReduceLROnPlateau, fixed_seed, load_parameters, train
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
    if model_type == "classifier":
        model = FAN(num_HG=num_HG)
    elif model_type == "regressor":
        model = RegressionModel(backbone, dropout=dropout)
    # Create train/val set
    print("Loading annotation...")
    train_set, val_set = get_train_val_dataset(data_root=train_data_root, 
                                               annot_path=train_annot, 
                                               train_size=split_ratio,
                                               use_image_ratio=use_image_ratio,
                                               model_type=model_type)
    print("End of Loading annotation!!!")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    # Optimizer adn criterion
    if model_type == "classifier":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=0.9, 
                                        weight_decay=1e-6)
        criterion = nn.MSELoss(reduction="sum")

    elif model_type == "regressor":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=lr,
                                        momentum=0.9, 
                                        weight_decay=1e-6)
        criterion = nn.MSELoss()
    
    # Scheduler
    if scheduler_type == 0:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif scheduler_type == 1:
        warm_epoch = cfg['warm_epoch']
        scheduler = Warmup_ReduceLROnPlateau(optimizer, warm_epoch, patience=3, verbose=True)
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
        exp_name=exp_name,
        only_save_best=only_save_best,
        resume_epoch=resume_epoch)

    


if __name__ == '__main__':
    main()
