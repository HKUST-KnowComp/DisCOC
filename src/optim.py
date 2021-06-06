import math

from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps,
        num_schedule_steps,
        min_percent
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(LinearWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            self.min_percent,
            float(self.num_schedule_steps - current_step) / \
                float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        )
