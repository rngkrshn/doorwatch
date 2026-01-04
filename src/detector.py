import cv2
import numpy as np

class MotionDetector:
    def __init__(self, roi, diff_threshold=25, blur_ksize=11):
        self.roi = roi  # (x,y,w,h)
        self.diff_threshold = diff_threshold
        self.blur_ksize = blur_ksize
        self.prev = None

    def _crop(self, frame):
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]

    def _prep(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return gray

    def update(self, frame) -> float:
        roi_frame = self._crop(frame)
        gray = self._prep(roi_frame)

        if self.prev is None:
            self.prev = gray
            return 0.0

        diff = cv2.absdiff(self.prev, gray)
        _, th = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        motion_frac = float(np.mean(th > 0))  # 0..1
        self.prev = gray
        return motion_frac

