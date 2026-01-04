# src/main.py
from __future__ import annotations

import os
import time
import cv2

from src.detector import MotionDetector
from src.state import DoorStateConfig, DoorStateMachine
from src.notify import Notifier


# ---------------- CONFIG ----------------
CAM_INDEX = 0

FRAME_W, FRAME_H = 640, 480
FPS_SLEEP = 0.10  # ~10 FPS

# ROI = (x, y, w, h)
# Use your tuned ROI here (example below from your calibration run):
ROI = (360, 140, 120, 260)

DIFF_THRESHOLD = 25
BLUR_KSIZE = 11

WARMUP_SECONDS = 1.0

# State thresholds/debounce (tune as needed)
OPEN_THRESHOLD = 0.02
CLOSE_THRESHOLD = 0.008
OPEN_DEBOUNCE = 3
CLOSE_DEBOUNCE = 6

PRINT_STATUS_EVERY_SEC = 5.0

# Notifications
NOTIFY_ON_CLOSED = False
NOTIFY_MIN_INTERVAL_SEC = 30  # hard rate limit


# ---------------- CAMERA ----------------
def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    # Prefer MJPG (reduces USB bandwidth vs raw YUYV)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(0.5)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")
    return cap


def warmup(cap: cv2.VideoCapture, det: MotionDetector) -> None:
    start = time.time()
    while time.time() - start < WARMUP_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        det.update(frame)  # primes prev-frame and stabilizes exposure


# ---------------- MAIN ----------------
def main() -> None:
    cap = open_camera()

    det = MotionDetector(
        roi=ROI,
        diff_threshold=DIFF_THRESHOLD,
        blur_ksize=BLUR_KSIZE,
        closed_ref_path="data/closed_ref.jpg",
    )

    cfg = DoorStateConfig(
        open_threshold=OPEN_THRESHOLD,
        close_threshold=CLOSE_THRESHOLD,
        open_debounce=OPEN_DEBOUNCE,
        close_debounce=CLOSE_DEBOUNCE,
    )
    sm = DoorStateMachine(cfg)

    # Notifier reads PUSHOVER_* from .env (via your src/notify.py)
    notifier = Notifier(min_interval_sec=NOTIFY_MIN_INTERVAL_SEC)

    warmup(cap, det)

    print("Running doorwatch. Ctrl+C to stop.")
    last_status = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            out = det.update(frame)

            # State machine should use delta vs CLOSED reference, not motion
            event = sm.step(out.delta_closed_frac)

            if event:
                # Print event
                print(
                    f"[EVENT] {event} (delta_closed={out.delta_closed_frac:.4f}, motion={out.motion_frac:.4f}) state={sm.state}"
                )

                # Notify only on transitions (Notifier itself rate-limits)
                if event == "OPEN":
                    notifier.send(
                        title="DoorWatch",
                        message=f"Door OPEN (delta={out.delta_closed_frac:.3f})",
                    )
                elif event == "CLOSED" and NOTIFY_ON_CLOSED:
                    notifier.send(
                        title="DoorWatch",
                        message=f"Door CLOSED (delta={out.delta_closed_frac:.3f})",
                    )

            if time.time() - last_status >= PRINT_STATUS_EVERY_SEC:
                print(
                    f"[STATUS] delta_closed={out.delta_closed_frac:.4f} motion={out.motion_frac:.4f} state={sm.state}"
                )
                last_status = time.time()

            time.sleep(FPS_SLEEP)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("Stopped.")


if __name__ == "__main__":
    main()
