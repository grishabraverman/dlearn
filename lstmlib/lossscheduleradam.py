import logging

from torch.optim import Optimizer

from lstmlib.lossschedulerbase import LossSchedulerBase

log = logging.getLogger(__name__)


class LossSchedulerAdam(LossSchedulerBase):
    def __init__(self, optimizer: Optimizer, factor_on_divergence: float,
                 min_learn_rate: float, loss_min_change_perc: float,
                 max_stuck_events: int):
        super().__init__(loss_min_change_perc, max_stuck_events)
        self.factor_on_divergence = factor_on_divergence
        self.optimizer = optimizer
        self.min_learn_rate = min_learn_rate
        return

    def step(self, loss: float):
        first_time = self.current_loss is None
        super().step(loss)
        if first_time:
            return
        if self.get_reset_flag():
            learning_rate = self.optimizer.param_groups[0]['lr']
            new_learning_rate = max(learning_rate * self.factor_on_divergence, self.min_learn_rate)
            log.info("Adam plateau; reducing learning rate to " + str(new_learning_rate))
            self.optimizer.param_groups[0]['lr'] = new_learning_rate
        return
