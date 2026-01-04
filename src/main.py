# src/main.py
import time
import cv2

from .detector import MotionDetector
from .state import DoorStateConfig, DoorStateMachine

# ---------- Camera / runtime config ----------
CAM_INDEX = 0
W, H = 640, 480

# ROI = (x, y, width, height)
# Start full-frame. Next step: tighten this to just the door edge/handle.
ROI = (0, 0, W, H)

# Your camera supports YUYV at 640x480 @ 30fps (per v4l2-ctl).
FOURCC = "YUYV"
TARGET_FPS = 30

# Warm up camera exposure/white-balance to reduce startup false positives
WARMUP_SECONDS = 2.0

# Loop timing
FPS_SLEEP = 1.0 / TARGET_FPS
# --------------------------------------------


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible (cap.isOpened() == False)")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    return cap


def warmup(cap: cv2.VideoCapture, det: MotionDetector) -> None:
    start = time.time()
    while time.time() - start < WARMUP_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        det.update(frame)  # prime detector prev-frame


def main():
    cap = open_camera()

    det = MotionDetector(roi=ROI, diff_threshold=25, blur_ksize=11)

    # Based on your calibration:
    # baseline ~0.002-0.004, door movement spikes 0.03+
    cfg = DoorStateConfig(
        open_threshold=0.02,
        close_threshold=0.008,
        open_debounce=5,   # reduces false OPENs from tiny spikes
        close_debounce=10, # requires stability before re-arming
    )
    sm = DoorStateMachine(cfg)

    warmup(cap, det)

    print("Running doorwatch. Ctrl+C to stop.")
    last_status = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            motion = det.update(frame)
            event = sm.step(motion)

            if event:
                print(f"[EVENT] {event} (motion={motion:.4f}) state={sm.state}")

            # periodic status every 5 seconds (useful while tuning ROI)
            if time.time() - last_status >= 5.0:
                print(f"[STATUS] motion={motion:.4f} state={sm.state}")
                last_status = time.time()

            time.sleep(FPS_SLEEP)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("Stopped.")


if __name__ == "__main__":
    main()
