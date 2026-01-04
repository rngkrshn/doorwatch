import cv2
import numpy as np
import time

# --------- CONFIG (edit these) ----------
CAM_INDEX = 0
W, H = 640, 480

# ROI = (x, y, width, height)
# Start with full frame. After first run, narrow this to just the door edge/handle area.
ROI = (360, 140, 120, 260)


# Motion metric settings
BLUR_KSIZE = 11          # odd number
DIFF_THRESHOLD = 25      # pixel threshold for "changed"
PRINT_EVERY = 10         # frames
# ----------------------------------------

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

time.sleep(1)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read initial frame")

prev = cv2.cvtColor(crop(frame, ROI), cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (BLUR_KSIZE, BLUR_KSIZE), 0)

frame_count = 0

print("Calibrating... Press Ctrl+C to stop.")
print(f"ROI={ROI}, DIFF_THRESHOLD={DIFF_THRESHOLD}, BLUR_KSIZE={BLUR_KSIZE}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(crop(frame, ROI), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

        diff = cv2.absdiff(prev, gray)
        _, th = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # motion score = fraction of pixels that changed
        motion_frac = float(np.mean(th > 0))  # 0.0 to 1.0

        frame_count += 1
        if frame_count % PRINT_EVERY == 0:
            print(f"motion_frac={motion_frac:.4f}")

        prev = gray
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    print("Done.")
