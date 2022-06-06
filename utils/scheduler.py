from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

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

    def step(self, metric=None):
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