import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation import heatmap_to_landmark, NME
import numpy as np
from tqdm import tqdm
import random
import os
import time

def mkdir_if_exist(path:str):
    if not os.path.isdir(path):
        os.mkdir(path)

def fixed_seed(myseed:int):
    """Initial random seed
    Args:
        myseed: the seed for initial
    """
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def load_parameters(model, path, optimizer=None, scheduler=None):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)

    if optimizer != None and scheduler!= None:
        optimizer.load_state_dict(torch.load(os.path.join(f"./save/optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(f"./save/scheduler.pt")))

    print("End of loading !!!")

def val(model, test_loader, device, fix_coord=False, add_boundary=False):
    print("Starting Validation....")
    
    model = model.to(device)
    
    total_NME_loss = 0
    total_NEM_loss_68 = np.zeros(68)
    num_data = 0

    model.eval()
    for sample in tqdm(test_loader):
        with torch.no_grad():
            img, gt_label = sample['img'], sample['gt_label']
            img = img.to(device)

            if add_boundary:
                outputs, pred_boundary = model(img)
            else:
                outputs = model(img)

            pred = heatmap_to_landmark(outputs,fix_coord=fix_coord)
            pred_loss, pred_loss_68 = NME(pred, gt_label, average=False, return_68=True)
            num_data += img.shape[0]
            total_NME_loss += pred_loss
            total_NEM_loss_68 += pred_loss_68.sum(axis=0)
    print("End of validating....")


    return (total_NME_loss / num_data), (total_NEM_loss_68 / num_data)

def process_loss(loss_type:str, criterion, outputs:torch.Tensor, label:torch.Tensor, weight_map:torch.Tensor=None):
    loss = 0
    if loss_type =="L2":
        num_target = (label != 0).sum()
        for output in outputs:
            loss += criterion(output, label) / num_target
    elif loss_type == "wing_loss":
        for output in outputs:
            loss += criterion(output, label)
    elif loss_type == "weighted_L2" or loss_type == "adaptive_wing_loss":
        if weight_map == None:
            raise ValueError("Weight map cannot be None!")
        for output in outputs:
            loss += criterion(output, label, weight_map)

    return loss

def process_boundary(loss_type:str, criterion, outputs:torch.Tensor, boundary:torch.Tensor, weight_map:torch.Tensor=None):
    loss = 0
    if loss_type =="L2":
        num_target = (boundary != 0).sum()
        for output in outputs:
            loss += criterion(output, boundary) / num_target
    elif loss_type == "wing_loss":
        for output in outputs:
            loss += criterion(output, boundary)
    elif loss_type == "weighted_L2" or loss_type == "adaptive_wing_loss":
        if weight_map == None:
            raise ValueError("Weight map cannot be None!")
        for output in outputs:
            loss += criterion(output, boundary, weight_map)

    return loss

def train(model, train_loader, val_loader, test_loader, epoch:int, save_path:str, device, criterion, scheduler, optimizer, 
        loss_type:str, exp_name="", train_hyp=dict(), resume_epoch=-1,fix_coord=False, add_boundary=False):
    start_train = time.time()
    # Create writerr for recording loss
    if exp_name == "":
        writer = SummaryWriter()
    else:
        path = f"./runs/"
        os.makedirs(path,exist_ok=True)
        writer = SummaryWriter(os.path.join(path, exp_name))

    # Create directory for saving model.
    os.makedirs(save_path,exist_ok=True)

    # Best NME loss and epoch for save best .pt
    best_val_NME_loss = 999
    best_test_NME_loss = 999
    best_val_epoch = 0
    best_test_epoch = 0

    use_weight_map = (loss_type == "weighted_L2") or (loss_type == "adaptive_wing_loss")

    # Set start epoch and end epoch
    if resume_epoch != -1:
        print("Resume training!!")
        start_epoch = resume_epoch
        end_epoch = resume_epoch + epoch
    else:
        start_epoch = 1
        end_epoch = epoch + 1

    # Starting training
    for epoch in range(start_epoch, end_epoch):
        print(f'epoch = {epoch}')
        start_time = time.time()


        # Record learning rate
        writer.add_scalar(tag="train/learning_rate",
                            scalar_value=float(optimizer.param_groups[0]['lr']), 
                            global_step=epoch)
        
        # Training part
        model.train()
        train_loss = 0.0
        for sample in tqdm(train_loader):
            img = sample['img']
            label = sample['label']

            # Add weight map
            weight_map = None
            if use_weight_map:
                weight_map = sample['weight_map'].to(device)
                
            # Add boundary
            if add_boundary:
                boundary = sample['boundary']

            # Forward part
            img = img.to(device)
            label = label.to(device)
            if add_boundary:
                
                outputs, pred_boundary = model(img)
            else:
                outputs = model(img)
            # Calculate Loss
            loss = process_loss(loss_type, criterion, outputs, label, weight_map)
            if add_boundary:
                boundary_loss = process_boundary(loss_type, criterion, pred_boundary, boundary, weight_map)
                loss += boundary_loss
            # Backward and update
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            optimizer.step()
            scheduler.step()
            del loss
  
        # Recording Epoch loss with tensorboard
        train_loss /= len(train_loader.dataset)
        writer.add_scalar(tag="train/loss",
                        scalar_value=float(train_loss), 
                        global_step=epoch)

        
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_NME_loss = 0.0
            for sample in tqdm(val_loader):
                img, label, gt_label = sample['img'], sample['label'], sample['gt_label']

                # Weight map
                weight_map = None
                if use_weight_map:
                    weight_map = sample['weight_map'].to(device)
                # Add boundary
                if add_boundary:
                    boundary = sample['boundary']
                # Forward part
                img = img.to(device)
                label = label.to(device)
                if add_boundary:
                    outputs, pred_boundary = model(img)
                else:
                    outputs = model(img)
                # intermediate supervision
                loss = process_loss(loss_type, criterion, outputs, label, weight_map)
                if add_boundary:
                    boundary_loss = process_boundary(loss_type, criterion, pred_boundary, boundary, weight_map)
                    loss += boundary_loss
                val_loss += loss.item()
                # Calculate loss with groud truth label
                pred_label = heatmap_to_landmark(outputs, fix_coord=fix_coord)

                val_NME_loss += NME(pred_label, gt_label)

            
            val_loss /= len(val_loader.dataset)
            val_NME_loss /= len(val_loader.dataset)
            writer.add_scalar(tag="val/NME_loss",
                            scalar_value=float(val_NME_loss), 
                            global_step=epoch)
            writer.add_scalar(tag="val/loss",
                            scalar_value=float(val_loss), 
                            global_step=epoch)
        # Testing part
        test_NME_loss, test_NME_loss_68 = val(model, test_loader, device, fix_coord=fix_coord, add_boundary=add_boundary)
        writer.add_scalar(tag="val/test_NME_loss",
                                scalar_value=float(test_NME_loss), 
                                global_step=epoch)
        # Scheduler steping
        scheduler.step(val_NME_loss)

        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        print('='*24)
        print('time = {} MIN {:.1f} SEC, total time = {} Min {:.1f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        formatted_str = "{: <20} : {:.6f}"
        print(formatted_str.format('Training loss', train_loss))
        print(formatted_str.format('Validating loss', val_loss))
        print(formatted_str.format('Validating NME loss', val_NME_loss))
        print(formatted_str.format('Testing NME loss', test_NME_loss))
        print('='*24 + '\n')


        if test_NME_loss < best_test_NME_loss:
            best_test_NME_loss = test_NME_loss
            best_test_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
        if val_NME_loss < best_val_NME_loss:
            best_val_NME_loss = val_NME_loss
            best_val_epoch = epoch
        
        # Save model, scheduler and optimizer
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(save_path, f'scheduler.pt'))

    
    print("End of training !!!")
    print(f"Best validating NME loss {best_val_NME_loss:.6f} on epoch {best_val_epoch}")
    print(f"Best testing NME loss {best_test_NME_loss:.6f} on epoch {best_test_epoch}")


    metric_result = dict()
    metric_result['Best/val_NME_loss'] = best_val_NME_loss
    metric_result['Best/val_epoch'] = best_val_epoch
    metric_result['Best/test_NME_loss'] = best_test_NME_loss
    metric_result['Best/test_epoch'] = best_test_epoch

    writer.add_hparams(train_hyp, metric_result)

    writer.close()