# src/state.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class DoorStateConfig:
    open_threshold: float = 0.05
    close_threshold: float = 0.02
    open_debounce: int = 3
    close_debounce: int = 6


class DoorStateMachine:
    """
    Uses delta_closed (difference from closed reference):
      - OPEN when delta_closed stays above open_threshold for open_debounce frames
      - CLOSED when delta_closed stays below close_threshold for close_debounce frames

    Emits exactly one event per transition: "OPEN" or "CLOSED".
    """
    def __init__(self, cfg: DoorStateConfig):
        self.cfg = cfg
        self.state = "CLOSED"  # forced initial state (prevents "OPEN at startup" spam)
        self._open_count = 0
        self._close_count = 0

    def step(self, delta_closed: float) -> Optional[str]:
        if self.state == "CLOSED":
            if delta_closed >= self.cfg.open_threshold:
                self._open_count += 1
            else:
                self._open_count = 0

            if self._open_count >= self.cfg.open_debounce:
                self.state = "OPEN"
                self._open_count = 0
                self._close_count = 0
                return "OPEN"

        else:  # OPEN
            if delta_closed <= self.cfg.close_threshold:
                self._close_count += 1
            else:
                self._close_count = 0

            if self._close_count >= self.cfg.close_debounce:
                self.state = "CLOSED"
                self._open_count = 0
                self._close_count = 0
                return "CLOSED"

        return None
