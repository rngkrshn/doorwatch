# src/main.py
import os
import time

import cv2

from src.detector import MotionDetector
from src.notify import Notifier
from src.state import DoorStateConfig, DoorStateMachine

# ---------------- CONFIG ----------------
CAM_INDEX = 0
W, H = 640, 480

# ROI = (x, y, width, height)
# Use your tuned ROI here:
ROI = (360, 140, 120, 260)

# MotionDetector thresholds
DIFF_THRESHOLD = 25
BLUR_KSIZE = 11

# Self-calibration for closed reference:
# Requires the ROI to be stable for N frames; then captures "closed_ref" automatically.
CAL_STABLE_MOTION_THRESHOLD = 0.002
CAL_STABLE_REQUIRED_FRAMES = 20
CAL_MAX_SECONDS = 20.0

# State machine thresholds (tune)
CFG = DoorStateConfig(
    open_threshold=0.04,   # raise/lower based on your delta_closed
    close_threshold=0.015,
    open_debounce=3,
    close_debounce=8,
)

# Loop timing
FPS_SLEEP = 0.08  # ~12.5 FPS
STATUS_EVERY_SEC = 5.0

# Notifications
NOTIFY_ON_CLOSED = os.getenv("PUSHOVER_NOTIFY_ON_CLOSED", "0").strip() in ("1", "true", "True")

# ---------------------------------------


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Prefer MJPG if supported (lower CPU on many USB cams)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Reduce latency if supported
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    time.sleep(0.5)
    return cap


def warmup(cap: cv2.VideoCapture, det: MotionDetector, seconds: float = 1.0) -> None:
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        det.update(frame)  # primes prev frame and helps calibration


def main() -> None:
    cap = open_camera()

    det = MotionDetector(
        roi=ROI,
        diff_threshold=DIFF_THRESHOLD,
        blur_ksize=BLUR_KSIZE,
        closed_ref_gray=None,  # self-calibrate
        stable_motion_threshold=CAL_STABLE_MOTION_THRESHOLD,
        stable_required_frames=CAL_STABLE_REQUIRED_FRAMES,
        max_calibration_seconds=CAL_MAX_SECONDS,
    )

    sm = DoorStateMachine(CFG)
    notifier = Notifier(min_interval_sec=30)

    warmup(cap, det, seconds=1.0)

    print("Running doorwatch. Ctrl+C to stop.")
    last_status = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            motion, delta_closed = det.update(frame)

            # If not calibrated yet, just show status and wait.
            if not det.calibrated:
                if time.time() - last_status >= STATUS_EVERY_SEC:
                    print(f"[CALIBRATING] motion={motion:.4f} (waiting for still ROI...)")
                    last_status = time.time()
                time.sleep(FPS_SLEEP)
                continue

            event = sm.step(delta_closed)

            if event == "OPEN":
                print(f"[EVENT] OPEN (delta_closed={delta_closed:.4f}, motion={motion:.4f}) state={sm.state}")
                notifier.send("Doorwatch", "Door opened")
            elif event == "CLOSED":
                print(f"[EVENT] CLOSED (delta_closed={delta_closed:.4f}, motion={motion:.4f}) state={sm.state}")
                if NOTIFY_ON_CLOSED:
                    notifier.send("Doorwatch", "Door closed")

            if time.time() - last_status >= STATUS_EVERY_SEC:
                print(f"[STATUS] delta_closed={delta_closed:.4f} motion={motion:.4f} state={sm.state}")
                last_status = time.time()

            time.sleep(FPS_SLEEP)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("Stopped.")


if __name__ == "__main__":
    main()
