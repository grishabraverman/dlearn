import logging

log = logging.getLogger(__name__)

class LossSchedulerBase:
    def __init__(self, loss_min_change_perc: float, max_stuck_events: int):
        self.loss_min_change_perc = loss_min_change_perc
        self.max_stuck_events = max_stuck_events
        self.current_loss = None
        self.stuck_events_counter = 0
        self.loss_stuck_flag = False

    def step(self, loss: float):
        if self.current_loss is None:
            self.current_loss = loss
            return
        loss_change = abs(self.current_loss - loss) / loss * 100.0
        self.stuck_status(loss_change)
        self.current_loss = loss
        return

    def stuck_status(self, loss_change_perc):
        if loss_change_perc < self.loss_min_change_perc:
            self.loss_stuck_flag = True
            log.info("Stuck flag is raised")
            self.stuck_events_counter = self.stuck_events_counter + 1
            log.info("Stuck event counter " + str(self.stuck_events_counter))
        else:
            self.loss_stuck_flag = False

    def get_reset_flag(self):
        return self.loss_stuck_flag

    def is_stuck_finally(self):
        return self.stuck_events_counter >= self.max_stuck_events
