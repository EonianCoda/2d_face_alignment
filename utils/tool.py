
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
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


class Warmup_ReduceLROnPlateau(_LRScheduler):
    def __init__(self, optimizer, warm_up_epoch, patience=3, verbose=True):
        self.warm_up_epoch = warm_up_epoch
        self.cur_epoch = 0
        self.after_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        self.verbose = verbose
        super().__init__(optimizer)
        self.last_epoch = 0

    def get_lr(self) -> float:
        if self.last_epoch < self.warm_up_epoch:
            return [base_lr * (float(self.last_epoch + 1) / self.warm_up_epoch) for base_lr in self.base_lrs]
        else:
            return self.after_scheduler._last_lr

    def _print_info(self):
        for group_idx, lr in enumerate(self.get_lr()):
            print("Epoch {:4d}: Adjusting learning rate of group {} to {:4e}.".format(self.last_epoch, group_idx, lr))

    def step(self,metric=None):
        cur_lr = self.get_lr()
        self.last_epoch = 1 if self.last_epoch == 0 else self.last_epoch + 1
        if self.last_epoch < self.warm_up_epoch:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        elif metric != None:
            self.after_scheduler.step(metric)
        else:
            raise ValueError("Currnet epoch is larger than warm up epoch and metirc still is None")
        if self.verbose and cur_lr != self.get_lr():
            self._print_info()

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)
    print("End of loading !!!")


def val(model, test_loader, device):
    print("Starting Validation....")
    
    model = model.to(device)
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    total_NME_loss = 0
    num_data = 0
    for batch_idx, (data, label, gt_label) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data = data.to(device)
            # label = label.to(device)
            outputs = model(data)
            # loss = 0
            # for output in outputs:
            #     loss += criterion(output, label)
            pred = heatmap_to_landmark(outputs)
            pred_loss = NME(pred, gt_label, average=False)
            num_data += data.shape[0]
            total_NME_loss += pred_loss
    print("End of validating....")
    print(f"Average NME Loss : {total_NME_loss / num_data:.4f}")
    return total_NME_loss / num_data

def train(model, train_loader, val_loader, test_loader, epoch:int, save_path:str, device, criterion, scheduler, optimizer, exp_name="", only_save_best=False):
    start_train = time.time()
    
    overall_loss = []
    overall_val_loss = []
    if exp_name == "":
        writer = SummaryWriter()
    else:
        writer = SummaryWriter("./runs/" + exp_name)
    global_training_step = 0
    global_validation_step = 0
    # Create directory for saving model
    mkdir_if_exist("./save")
    # Best NME loss and epoch for save best .pt
    best_val_NME_loss = 999
    best_epoch = 0
    # Starting training
    for epoch in range(1, epoch + 1):
        print(f'epoch = {epoch}')
        start_time = time.time()
        train_loss = 0.0
        train_NME_loss = 0.0
        # Training part
        model.train()

        writer.add_scalar(tag="hyperparameters/lr",
                scalar_value=float(optimizer.param_groups[0]['lr']), 
                global_step=epoch)
        for batch_idx, (data, label, gt_label) in enumerate(tqdm(train_loader)):
            
            data = data.to(device)
            label = label.to(device)
            outputs = model(data) 
            # intermediate supervision
            loss = 0
            num_target = (label != 0).sum()
            for output in outputs:
                loss += criterion(output, label) / num_target
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            optimizer.step()

            writer.add_scalar(tag="train/step_loss",
                            scalar_value=float(loss),
                            global_step=global_training_step)
            
            # Calculate gt loss with groud truth label
            pred_label = heatmap_to_landmark(outputs)
            NME_loss = NME(pred_label, gt_label)
            writer.add_scalar(tag="train/NME_step_loss",
                            scalar_value=float(NME_loss), 
                            global_step=global_training_step)
            train_NME_loss += NME_loss
            global_training_step += 1
            del loss
  
        train_loss /= len(train_loader.dataset)
        train_NME_loss /= len(train_loader.dataset)

        writer.add_scalar(tag="train/epoch_loss",
                        scalar_value=float(train_loss), 
                        global_step=epoch)
        writer.add_scalar(tag="train/NME_epoch_loss",
                        scalar_value=float(train_NME_loss), 
                        global_step=epoch)
        overall_loss.append(float(train_loss))
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_NME_loss = 0.0
            for batch_idx, (data, label, gt_label) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                label = label.to(device)

                outputs = model(data)
                # intermediate supervision
                loss = 0
                num_target = (label != 0).sum()
                for output in outputs:
                    loss += criterion(output, label) / num_target
                val_loss += loss.item()
                writer.add_scalar(tag="val/step_loss",
                            scalar_value=float(loss), 
                            global_step=global_validation_step)

                # Calculate loss with groud truth label
                pred_label = heatmap_to_landmark(outputs)
                NME_loss =  NME(pred_label, gt_label)
                writer.add_scalar(tag="val/NME_step_loss",
                                scalar_value=float(NME_loss), 
                                global_step=global_validation_step)
                val_NME_loss += NME_loss
                global_validation_step += 1
            
            val_loss /= len(val_loader.dataset)
            val_NME_loss /= len(val_loader.dataset)
            writer.add_scalar(tag="val/NME_epoch_loss",
                scalar_value=float(val_NME_loss), 
                global_step=epoch)
            writer.add_scalar(tag="val/epoch_loss",
                            scalar_value=float(val_loss), 
                            global_step=epoch)
            overall_val_loss.append(float(val_loss))
        # Testing part
        test_NME_loss = val(model, test_loader, device)
        writer.add_scalar(tag="test/NME_loss",
                                scalar_value=float(test_NME_loss), 
                                global_step=epoch)
        # Scheduler steping
        scheduler.step(val_loss)

        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        formatted_str = "{: <20} : {:.6f}"
        print(formatted_str.format('Training loss', train_loss))
        print(formatted_str.format('Training NME loss', train_NME_loss))
        print(formatted_str.format('Validating loss', val_loss))
        print(formatted_str.format('Validating NME loss', val_NME_loss))
        print(formatted_str.format('Testing NME loss', test_NME_loss))
        print('========================\n')

        if val_NME_loss < best_val_NME_loss:
            best_val_NME_loss = val_NME_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
        if not only_save_best:
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))

    writer.close()
    print("End of training !!!")
    print(f"Best Epoch = {best_epoch}")
    print(f"Best val NEM loss = {best_val_NME_loss:.6f}")