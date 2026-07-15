import logging
import math

from torch.optim import Optimizer

from lstmlib.lossschedulerbase import LossSchedulerBase

log = logging.getLogger(__name__)

class LossSchedulerSGD(LossSchedulerBase):
    def __init__(self, optimizer: Optimizer, factor_on_improvements: float, factor_on_divergence: float,
                 min_learn_rate: float, default_learn_rate: float, loss_min_change_perc: float,
                 max_stuck_events: int):
        super().__init__(loss_min_change_perc, max_stuck_events)
        self.factor_on_improvements = factor_on_improvements
        self.factor_on_divergence = factor_on_divergence
        self.optimizer = optimizer
        self.min_learn_rate = min_learn_rate
        self.default_learn_rate = default_learn_rate
        return

    def step(self, loss: float):
        first_time = self.current_loss is None
        prev_loss = self.current_loss
        super().step(loss)
        if first_time:
            return
        if prev_loss is None or not math.isfinite(loss) or loss == 0.0:
            log.warning("Skipping LR update due to invalid loss: " + str(loss))
            return
        learning_rate = self.optimizer.param_groups[0]['lr']
        factor = prev_loss / loss
        if loss < prev_loss:
            factor = factor * self.factor_on_improvements
        else:
            factor = factor * self.factor_on_divergence
        new_learning_rate = learning_rate * factor
        log.info("New learning rate: " + str(new_learning_rate))
        reset_learn_rate = self.get_reset_flag()
        if new_learning_rate < self.min_learn_rate or reset_learn_rate:
            log.info("Learning rate is minimal or training is stuck, resetting to default learning rate")
            new_learning_rate = self.default_learn_rate
        self.optimizer.param_groups[0]['lr'] = new_learning_rate
        return
