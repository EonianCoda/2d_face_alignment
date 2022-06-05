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

def load_parameters(model, path, optimizer=None, scheduler=None, model_type:str=""):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)

    if optimizer != None and scheduler!= None:
        if model_type == "":
            raise ValueError("If optimizer and scheduler are given, then model_type cannot be empty!")
        optimizer.load_state_dict(torch.load(os.path.join(f"./save/{model_type}", "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(f"./save/{model_type}", "scheduler.pt")))

    print("End of loading !!!")

def val(model, test_loader, device, model_type:str,fix_coord=False):
    print("Starting Validation....")
    
    model = model.to(device)
    
    total_NME_loss = 0
    total_NEM_loss_68 = np.zeros(68)
    num_data = 0

    model.eval()
    for data in tqdm(test_loader):
        with torch.no_grad():
            img, _ , gt_label = data
            img = img.to(device)

            outputs = model(img)

            if model_type == "classifier":
                pred = heatmap_to_landmark(outputs,fix_coord=fix_coord)
            elif model_type == "regressor":
                pred = outputs.detach().cpu()

            pred_loss, pred_loss_68 = NME(pred, gt_label, average=False, return_68=True)
            num_data += img.shape[0]
            total_NME_loss += pred_loss
            total_NEM_loss_68 += pred_loss_68.sum(axis=0)
    print("End of validating....")


    return (total_NME_loss / num_data), (total_NEM_loss_68 / num_data)

def process_loss(model_type:str, loss_type:str, criterion, outputs:torch.Tensor, label:torch.Tensor, weight_map:torch.Tensor=None):
    loss = 0
    if model_type == "classifier":
        if loss_type =="L2":
            num_target = (label != 0).sum()
            for output in outputs:
                loss += criterion(output, label) / num_target
        elif loss_type == "wing_loss":
            for output in outputs:
                loss += criterion(output, label)
        elif loss_type == "weighted_L2":
            if weight_map == None:
                raise ValueError("Weight map cannot be None!")
            for output in outputs:
                loss += criterion(output, label, weight_map)
        elif loss_type == "adaptive_wing_loss":
            for output in outputs:
                loss += criterion(output, label, weight_map)
    elif model_type == "regressor":
        loss += criterion(outputs, label)
    return loss
def train(model, train_loader, val_loader, test_loader, epoch:int, save_path:str, device, criterion, scheduler, optimizer, 
        model_type:str, loss_type:str, exp_name="", train_hyp:dict=dict(), only_save_best=False, resume_epoch=-1,fix_coord=False):
    start_train = time.time()
    # Create writerr for recording loss
    if exp_name == "":
        writer = SummaryWriter()
    else:
        path = f"./runs/{model_type}"
        os.makedirs(path,exist_ok=True)
        writer = SummaryWriter(os.path.join(path, exp_name))

    # Create directory for saving model.
    os.makedirs(save_path,exist_ok=True)

    # Best NME loss and epoch for save best .pt
    best_val_NME_loss = 999
    best_test_NME_loss = 999
    best_val_epoch = 0
    best_test_epoch = 0

    use_weight_map = (loss_type == "weighted_L2")

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
        writer.add_scalar(tag="hyperparameters/lr",
                            scalar_value=float(optimizer.param_groups[0]['lr']), 
                            global_step=epoch)
        
        # Training part
        model.train()
        train_loss = 0.0
        for data in tqdm(train_loader):
            if use_weight_map:
                img, label, weight_map = data
                weight_map = weight_map.to(device)
            else:
                img, label = data
                weight_map = None

            # Forward part
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            # Calculate Loss
            loss = process_loss(model_type, loss_type, criterion, outputs, label, weight_map)

            # Backward and update
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            optimizer.step()
         
            del loss
  
        # Recording Epoch loss with tensorboard
        train_loss /= len(train_loader.dataset)
        writer.add_scalar(tag="train/epoch_loss",
                        scalar_value=float(train_loss), 
                        global_step=epoch)

        
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_NME_loss = 0.0
            for data in tqdm(val_loader):
                if use_weight_map:
                    img, label, gt_label, weight_map = data
                    weight_map = weight_map.to(device)
                else:
                    img, label, gt_label = data
                    weight_map = None

                # Forward part
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)

                # intermediate supervision
                loss = process_loss(model_type, loss_type, criterion, outputs, label, weight_map)
                val_loss += loss.item()
                # Calculate loss with groud truth label
                if model_type == "classifier":
                    pred_label = heatmap_to_landmark(outputs, fix_coord=fix_coord)
                elif model_type == "regressor":
                    pred_label = outputs.detach().cpu()

                val_NME_loss += NME(pred_label, gt_label)

            
            val_loss /= len(val_loader.dataset)
            val_NME_loss /= len(val_loader.dataset)
            writer.add_scalar(tag="val/NME_epoch_loss",
                            scalar_value=float(val_NME_loss), 
                            global_step=epoch)
            writer.add_scalar(tag="val/epoch_loss",
                            scalar_value=float(val_loss), 
                            global_step=epoch)
        # Testing part
        test_NME_loss, test_NME_loss_68 = val(model, test_loader, device, model_type, fix_coord=fix_coord)
        writer.add_scalar(tag="test/NME_loss",
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
        if val_NME_loss < best_val_NME_loss:
            best_val_NME_loss = val_NME_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
        if not only_save_best:
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        # Save scheduler and optimizer
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(save_path, f'scheduler.pt'))

    
    print("End of training !!!")
    print(f"Best validating NME loss {best_val_NME_loss:.6f} on epoch {best_val_epoch}")
    print(f"Best testing NME loss {best_test_NME_loss:.6f} on epoch {best_test_epoch}")

    # metric_result = dict()
    # metric_result['best/val_NME_loss'] = best_val_NME_loss
    # metric_result['best/val_epoch'] = best_val_epoch
    # metric_result['best/test_NME_loss'] = best_test_NME_loss
    # metric_result['best/test_epoch'] = best_test_epoch
    # metric_result['end epoch'] = end_epoch

    # writer.add_hparams(train_hyp, metric_result)

    writer.close()