from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import math

class Warmup_ReduceLROnPlateau(_LRScheduler):
    def __init__(self, optimizer, warm_up_step, patience=3, verbose=True):
        self.warm_up_step = warm_up_step
        self.cur_step = 0
        self.after_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        self.verbose = verbose
        
        self.last_step = 0
        self.ratio = self.warm_up_step / 20
        self.flag = False
        super(Warmup_ReduceLROnPlateau, self).__init__(optimizer)
    def get_lr(self) -> float:
        if self.last_step <= self.warm_up_step:
            cur_ratio = math.ceil(self.last_step / self.ratio) / (self.warm_up_step / self.ratio)
            cur_ratio = max(cur_ratio, 0.5 / (self.warm_up_step / self.ratio))
            return [base_lr * cur_ratio for base_lr in self.base_lrs]
        else:

            return self.after_scheduler.get_last_lr()
    def _print_info(self):
        for group_idx, lr in enumerate(self.get_lr()):
            print("step {:4d}: Adjusting learning rate of group {} to {:4e}.".format(self.last_step, group_idx, lr))

    def step(self,metric=None):
        #cur_lr = self.get_lr()
        if self.last_step < self.warm_up_step:
            self.last_step = 1 if self.last_step == 0 else self.last_step + 1
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        elif metric != None:
            self.after_scheduler.step(metric)
