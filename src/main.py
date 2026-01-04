import time
import cv2
import numpy as np

from src.detector import MotionDetector
from src.state import DoorStateConfig, DoorStateMachine

# ---------- CONFIG ----------
W, H = 640, 480
CAM_INDEX = 0

DIFF_THRESHOLD = 25
BLUR_KSIZE = 11

FPS_SLEEP = 0.10  # ~10 FPS
# ---------------------------


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(1.0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")
    return cap


def preprocess_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if BLUR_KSIZE and BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return gray


def auto_roi_from_frame(frame):
    """
    Heuristic ROI selection:
    - Find strong vertical edges (door seams are vertical)
    - Pick the most edge-dense vertical strip
    - Return a reasonable ROI around it
    """
    gray = preprocess_gray(frame)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(gx)

    col_score = mag.mean(axis=0)  # shape (W,)

    # Smooth scores to avoid noisy peaks
    k = 25
    col_score_s = np.convolve(col_score, np.ones(k) / k, mode="same")

    x_peak = int(np.argmax(col_score_s))

    # ROI size (tune if needed)
    roi_w = 120
    roi_h = 260

    # Center ROI around x_peak
    x = max(0, min(W - roi_w, x_peak - roi_w // 2))

    # Vertical placement (tune if handle/edge is higher/lower)
    y = 140
    y = max(0, min(H - roi_h, y))

    return (x, y, roi_w, roi_h)


def capture_closed_reference(cap, roi, seconds=2.0):
    """
    Assumes the door is CLOSED at startup.
    Captures multiple ROI frames and averages them to reduce noise.
    """
    x, y, w, h = roi
    frames = []
    start = time.time()

    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue

        roi_bgr = frame[y : y + h, x : x + w]
        gray = preprocess_gray(roi_bgr)
        frames.append(gray.astype(np.float32))

        time.sleep(0.05)

    if len(frames) < 5:
        raise RuntimeError("Not enough frames captured for closed reference.")

    ref = np.mean(frames, axis=0).astype(np.uint8)
    return ref


def main():
    cap = open_camera()

    # Grab one usable frame for ROI calibration
    frame = None
    for _ in range(60):
        ret, f = cap.read()
        if ret:
            frame = f
            break
        time.sleep(0.05)

    if frame is None:
        cap.release()
        raise RuntimeError("Could not read initial frame for calibration.")

    # Auto ROI
    roi = auto_roi_from_frame(frame)
    print(f"[CAL] ROI={roi}")

    # Capture closed reference (door should be closed and view unobstructed)
    closed_ref_gray = capture_closed_reference(cap, roi, seconds=2.0)
    print("[CAL] Captured closed reference.")

    # Detector compares current ROI to CLOSED reference + also returns motion
    det = MotionDetector(
        roi=roi,
        diff_threshold=DIFF_THRESHOLD,
        blur_ksize=BLUR_KSIZE,
        closed_ref_gray=closed_ref_gray,
    )

    # State machine should use delta-to-closed, not raw motion
    cfg = DoorStateConfig(
        open_threshold=0.03,
        close_threshold=0.01,
        open_debounce=3,
        close_debounce=10,
    )
    sm = DoorStateMachine(cfg)

    print("Running doorwatch. Ctrl+C to stop.")
    last_status = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            motion, delta_closed = det.update(frame)
            event = sm.step(delta_closed)

            if event:
                print(
                    f"[EVENT] {event} (delta={delta_closed:.4f}, motion={motion:.4f}) state={sm.state}"
                )

            if time.time() - last_status >= 5.0:
                print(
                    f"[STATUS] delta={delta_closed:.4f} motion={motion:.4f} state={sm.state}"
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

