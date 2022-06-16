# Torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
# Model
from model.tool import get_model
from dataset.tool import get_train_val_dataset, get_test_dataset
from utils.tool import fixed_seed, load_parameters, train
from utils.scheduler import Warmup_MultiStepDecay
from losses.wing_loss import Adaptive_Wing_Loss
from losses.weighted_L2 import Weighted_L2
from cfg import *

import argparse

def add_weight_decay(net, l2_value):
    """no weight decay on bias and normalization layer
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        # skip bias and bn layer
        if name.endswith(".bias") or ("_bn" in name): # or ("norm" in name):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": l2_value},
    ]

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
    val_data_root = cfg['val_data_root']
    val_annot = cfg['val_annot']
    use_image_ratio = args.use_image_ratio

    ## Data setting ###
    batch_size =  cfg['batch_size']
    update_batch_size = cfg['update_batch_size']
    every_step_update = max(update_batch_size // batch_size, 1)
    split_ratio = cfg['split_ratio']
    aug_setting = cfg['aug_setting']
    bg_negative = cfg['bg_negative']
    ### Training hyperparameter ###
    epoch = cfg['epoch']
    seed = cfg['seed']
    lr = cfg['lr']
    ### Opitmizer Setting
    weight_decay = cfg['weight_decay']
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
  
    train_set, val_set = get_train_val_dataset(data_root=train_data_root, 
                                            annot_path=train_annot, 
                                            train_size=split_ratio,
                                            use_image_ratio=use_image_ratio,
                                            use_weight_map=use_weight_map,
                                            fix_coord=fix_coord,
                                            bg_negative=bg_negative,
                                            aug_setting=aug_setting)
    print("End of Loading annotation!!!")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers= 2, pin_memory=True, drop_last=True)

    params = add_weight_decay(model, weight_decay)
    # Optimizer
    optimizer = torch.optim.RMSprop(params,
                                    lr=lr,
                                    momentum=0.9)       

    # loss_type
    if loss_type == "L2":
        criterion = nn.MSELoss(reduction="sum")
    elif loss_type == "adaptive_wing_loss":
        criterion = Adaptive_Wing_Loss()
    elif loss_type == "weighted_L2":
        weight = cfg['weight']
        criterion = Weighted_L2(reduction="sum", weight=float(weight))
    
    # Scheduler
    warm_step = cfg['warm_step']
    milestones = cfg['milestones']
    milestones_lr = cfg['milestones_lr']
    scheduler = LambdaLR(optimizer, Warmup_MultiStepDecay(lr, warm_step, milestones, milestones_lr))
    model = model.to(device)
    
    # Testing data
    test_set = get_test_dataset(val_data_root, val_annot)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)


    # model save path
    save_path = f"./save/"

    # Resuming Training
    if resume:
        if resume_model_path == "":
            raise ValueError("If resume == True, then resume model path cannot be empty")
        load_parameters(model, resume_model_path, optimizer, epoch=resume_epoch - 1)
        scheduler.last_epoch = len(train_loader) * (resume_epoch - 1)
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        scheduler.step()


    # Print training information
    print("Start training!!\n")
    print(f"Loss type = {loss_type}")
    print(f"Length of training dataloader = {len(train_loader)}")
    print(f"Attention Block = {cfg['attention_blocks'][cfg['attention_block_idx']]}")
    print(f"Fix coord = {fix_coord}")
    print(f"Use CoordConv = {cfg['use_CoordConv']}")
    print(f"With_r = {cfg['with_r']}")
    print(f"Add CoordConv inHG = {cfg['add_CoordConv_inHG']}")
    print(f"Aug setting = {aug_setting}")
    print(f"Backgroud negative = {cfg['bg_negative']}")
    print(f"Batch Size = {cfg['batch_size']}")


    
    aug = [k for k, v in cfg['aug_setting'].items() if v]
    aug = " ".join(aug)
    loss_type = cfg['losses'][cfg['loss_idx']]
    train_hyp = {'loss_type': loss_type,
                'use_weight_map': (loss_type == "weighted_L2") or (loss_type == "adaptive_wing_loss"),
                'lr': cfg['lr'], 
                'use_image_ratio': use_image_ratio,
                'fix_coord': cfg['fix_coord'],
                'warm_step': cfg['warm_step'],
                'augmentation': aug,
                'seed': cfg['seed'],
                # model architecture
                'num_HG': cfg['num_HG'],
                'HG_depth': cfg['HG_depth'],
                'num_feats': cfg['num_feats'],
                'use_CoordConv':cfg['use_CoordConv'],
                'with_r': cfg['with_r'],
                'add_CoordConv_inHG':cfg['add_CoordConv_inHG'],
                'attention_block' : cfg['attention_blocks'][cfg['attention_block_idx']]}

    

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
        every_step_update=every_step_update,
        train_hyp=train_hyp,
        resume_epoch=resume_epoch)

    


if __name__ == '__main__':
    main()
