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
            self.milestones = None
            self.milestones_lr = None
    def __call__(self, cur_step:int):
        if cur_step <= self.warm_steps:
            return cal_warmup_ratio(cur_step, self.warm_steps)
        elif self.milestones != None:
            for i, step in enumerate(self.milestones):
                if cur_step >= step:
                    return float(self.milestones_lr[i] / self.base_lr)
        return 1.0
