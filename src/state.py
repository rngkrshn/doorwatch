from dataclasses import dataclass

@dataclass
class DoorStateConfig:
    open_threshold: float
    close_threshold: float
    open_debounce: int = 3
    close_debounce: int = 10

class DoorStateMachine:
    def __init__(self, cfg: DoorStateConfig):
        self.cfg = cfg
        self.state = "CLOSED"
        self._open_count = 0
        self._close_count = 0

    def step(self, delta_closed: float):
        if self.state == "CLOSED":
            if delta_closed >= self.cfg.open_threshold:
                self._open_count += 1
                if self._open_count >= self.cfg.open_debounce:
                    self.state = "OPEN"
                    self._open_count = 0
                    self._close_count = 0
                    return "OPEN"
            else:
                self._open_count = 0
            return None

        # state == "OPEN"
        if delta_closed <= self.cfg.close_threshold:
            self._close_count += 1
            if self._close_count >= self.cfg.close_debounce:
                self.state = "CLOSED"
                self._close_count = 0
                self._open_count = 0
                return "CLOSED"
        else:
            self._close_count = 0
        return None
