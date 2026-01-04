from dataclasses import dataclass

@dataclass
class DoorStateConfig:
    open_threshold: float = 0.02     # triggers OPEN when exceeded
    close_threshold: float = 0.008   # re-arms when below this
    open_debounce: int = 3           # consecutive frames above open_threshold
    close_debounce: int = 10         # consecutive frames below close_threshold

class DoorStateMachine:
    """
    States:
      - CLOSED: waiting for an "open" event
      - OPEN: waiting to re-arm after "close"
    """
    def __init__(self, cfg: DoorStateConfig):
        self.cfg = cfg
        self.state = "CLOSED"
        self.above_open = 0
        self.below_close = 0

    def step(self, motion_frac: float):
        event = None

        if self.state == "CLOSED":
            if motion_frac >= self.cfg.open_threshold:
                self.above_open += 1
            else:
                self.above_open = 0

            if self.above_open >= self.cfg.open_debounce:
                self.state = "OPEN"
                self.above_open = 0
                self.below_close = 0
                event = "OPEN"

        elif self.state == "OPEN":
            if motion_frac <= self.cfg.close_threshold:
                self.below_close += 1
            else:
                self.below_close = 0

            if self.below_close >= self.cfg.close_debounce:
                self.state = "CLOSED"
                self.below_close = 0
                event = "CLOSED"

        return event
