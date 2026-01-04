# src/notify.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class Notifier:
    def __init__(self, min_interval_sec: int = 30):
        self.token = os.getenv("PUSHOVER_APP_TOKEN")
        self.user = os.getenv("PUSHOVER_USER_KEY")

        if not self.token or not self.user:
            raise RuntimeError("Missing PUSHOVER_APP_TOKEN or PUSHOVER_USER_KEY")

        self.min_interval = min_interval_sec
        self._last_sent = 0.0

    def send(self, title: str, message: str) -> None:
        now = time.time()

        # hard rate-limit
        if now - self._last_sent < self.min_interval:
            return

        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.token,
                "user": self.user,
                "title": title,
                "message": message,
            },
            timeout=5,
        )

        if resp.status_code != 200:
            print(f"[WARN] Pushover failed: {resp.text}")
        else:
            self._last_sent = now

