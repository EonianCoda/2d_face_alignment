from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

class Warmup_ReduceLROnPlateau(_LRScheduler):
    def __init__(self, optimizer, warm_up_step, patience=3, verbose=True):
        self.warm_up_step = warm_up_step
        self.cur_epoch = 0
        self.after_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        self.verbose = verbose
        super().__init__(optimizer)
        self.last_step = 0

    def get_lr(self) -> float:
        if self.last_step < self.warm_up_step:
            return [base_lr * (float(self.last_step + 1) / self.warm_up_step) for base_lr in self.base_lrs]
        else:
            return self.after_scheduler._last_lr

    def _print_info(self):
        for group_idx, lr in enumerate(self.get_lr()):
            print("Epoch {:4d}: Adjusting learning rate of group {} to {:4e}.".format(self.last_step, group_idx, lr))

    @property
    def last_step(self):
        if self.last_step == 0:
            self.last_step = 1  
        else:
            self.last_step += 1
        return self.last_step

    def step(self):
        cur_lr = self.get_lr()
        if self.last_step < self.warm_up_step:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        if self.verbose and cur_lr != self.get_lr():
            self._print_info()

    def step_loss(self, metric):
        cur_lr = self.get_lr()
        if self.last_step >= self.warm_up_step:
            self.after_scheduler.step(metric)
        if self.verbose and cur_lr != self.get_lr():
            self._print_info()