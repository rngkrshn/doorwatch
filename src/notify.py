# src/notify.py
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class Notifier:
    """
    Pushover notifier.
    Required env:
      - PUSHOVER_APP_TOKEN
      - PUSHOVER_USER_KEY

    Optional env:
      - PUSHOVER_ENABLED=1 (default 1)
      - PUSHOVER_MIN_INTERVAL_SEC=30 (default 30)
      - PUSHOVER_NOTIFY_ON_CLOSED=0 (default 0)
    """

    def __init__(self, min_interval_sec: Optional[int] = None):
        enabled = os.getenv("PUSHOVER_ENABLED", "1").strip()
        self.enabled = enabled not in ("0", "false", "False", "")

        self.token = os.getenv("PUSHOVER_APP_TOKEN")
        self.user = os.getenv("PUSHOVER_USER_KEY")

        if not self.enabled:
            self.min_interval = 0
            self._last_sent = 0.0
            return

        if not self.token or not self.user:
            raise RuntimeError("Missing PUSHOVER_APP_TOKEN or PUSHOVER_USER_KEY in .env")

        if min_interval_sec is None:
            min_interval_sec = int(os.getenv("PUSHOVER_MIN_INTERVAL_SEC", "30"))

        self.min_interval = int(min_interval_sec)
        self._last_sent = 0.0

    def send(self, title: str, message: str) -> None:
        if not self.enabled:
            print(f"[NOTIFY disabled] {title}: {message}")
            return

        now = time.time()
        if now - self._last_sent < self.min_interval:
            # hard rate-limit
            return

        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.token,
                "user": self.user,
                "title": title,
                "message": message,
            },
            timeout=8,
        )

        if resp.status_code != 200:
            print(f"[WARN] Pushover failed ({resp.status_code}): {resp.text}")
            return

        self._last_sent = now


def main() -> None:
    n = Notifier()
    n.send("Doorwatch", "Test notification from Pi")


if __name__ == "__main__":
    main()
