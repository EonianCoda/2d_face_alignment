# from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, LambdaLR
# from torch.optim.lr_scheduler import MultiStepLR
import math

def cal_warmup_ratio(cur_step:int, num_steps:int, rise_type="step"):
    if rise_type == "step":
        step_length = num_steps / 20
        ratio = math.ceil(cur_step / step_length) / (num_steps / step_length)
        ratio = max(ratio, 0.5 / (num_steps / step_length))
        return ratio
    elif rise_type == "exp":
        ratio = 1.0 - math.exp((-1 * (cur_step + 1)) / num_steps)
        return ratio
class Warmup_MultiStepDecay(object):
    def __init__(self, base_lr:float, warm_steps:int, milestones=[], milestones_lr=[]):
        self.warm_steps = warm_steps
        self.base_lr = base_lr
        if milestones != [] and milestones_lr != []:
            if len(milestones_lr) != len(milestones):
                raise ValueError("Milestones_lr and milestones hould have same elements")
            elif warm_steps > milestones[0]:
                raise ValueError("Warm steps should less than milestone[0]")

            milestones.reverse()
            self.milestones = milestones
            milestones_lr.reverse()
            self.milestones_lr = milestones_lr
        else:
            milestones = None
            milestones_lr = None
        print(self.milestones_lr)
    def __call__(self, cur_step:int):
        if cur_step <= self.warm_steps:
            return cal_warmup_ratio(cur_step, self.warm_steps)
        elif self.milestones != None:
            for i, step in enumerate(self.milestones):
                if cur_step >= step:
                    return float(self.milestones_lr[i] / self.base_lr)
        return 1.0


# class Warmup_MultiStepDecay(_LRScheduler):
#     def __init__(self, optimizer, num_warm_steps, milestones=[]):
#         self.num_warm_steps = num_warm_steps
#         self.cur_step = 0
#         #self.milestones = milestones
       
#         # No weight decay
#         if milestones == []:
#             self.after_scheduler = None
#         else:
            
#             self.milestones = []
#             for step in milestones:
#                 self.milestones.append(step - num_warm_steps)

#             self.after_scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
#         super(Warmup_MultiStepDecay, self).__init__(optimizer)
    

#     def get_lr(self) -> float:
#         if self.cur_step <= self.num_warm_steps:
#             ratio = cat_warmup_ratio(self.cur_step, self.num_warm_steps)
#             return [base_lr * ratio for base_lr in self.base_lrs]
#         else:
#             return self.after_scheduler.get_last_lr()
#     def _print_info(self):
#         for group_idx, lr in enumerate(self.get_lr()):
#             print("step {:4d}: Adjusting learning rate of group {} to {:4e}.".format(self.cur_step, group_idx, lr))

#     def step(self):
#         if self.cur_step < self.num_warm_steps:
#             self.cur_step = 1 if self.cur_step == 0 else self.cur_step + 1
#             for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#                 param_group['lr'] = lr
#         elif self.after_scheduler != None:
#             self.after_scheduler.step()

# class Warmup_ReduceLROnPlateau(_LRScheduler):
#     def __init__(self, optimizer, num_warm_steps:int, patience=3, verbose=True):
#         self.num_warm_steps = num_warm_steps
#         #self.cur_step = 0
#         self.after_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
#         self.verbose = verbose
        
#         self.last_step = 0
#         # self.ratio = self.num_warm_steps / 20
#         #self.flag = False
#         super(Warmup_ReduceLROnPlateau, self).__init__(optimizer)
#     def get_lr(self) -> float:
#         if self.last_step <= self.num_warm_steps:
# #             # ExponentialWarmup
# # #            ratio = 1.0 - math.exp(-(self.last_step + 1) / self.num_warm_steps)
# #             cur_ratio = math.ceil(self.last_step / self.ratio) / (self.num_warm_steps / self.ratio)
# #             cur_ratio = max(cur_ratio, 0.5 / (self.num_warm_steps / self.ratio))
#             ratio = cat_warmup_ratio(self.last_step, self.num_warm_steps)
#             return [base_lr * ratio for base_lr in self.base_lrs]
#         else:
#             return self.after_scheduler.get_last_lr()
#     def _print_info(self):
#         for group_idx, lr in enumerate(self.get_lr()):
#             print("step {:4d}: Adjusting learning rate of group {} to {:4e}.".format(self.last_step, group_idx, lr))

#     def step(self,metric=None):
#         #cur_lr = self.get_lr()
#         if self.last_step < self.num_warm_steps:
#             self.last_step = 1 if self.last_step == 0 else self.last_step + 1
#             for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#                 param_group['lr'] = lr
#         elif metric != None:
#             self.after_scheduler.step(metric)

