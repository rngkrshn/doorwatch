import cv2
import numpy as np


class MotionDetector:
    """
    Returns (motion_frac, delta_closed_frac)
    delta_closed_frac compares current ROI to a fixed CLOSED reference ROI.
    """

    def __init__(self, roi, diff_threshold=25, blur_ksize=11, closed_ref_gray=None):
        self.roi = roi
        self.diff_threshold = diff_threshold
        self.blur_ksize = blur_ksize

        self.prev_gray = None

        if closed_ref_gray is None:
            raise RuntimeError("closed_ref_gray must be provided (self-calibration expected).")
        self.closed_ref_gray = closed_ref_gray

    def _crop(self, frame):
        x, y, w, h = self.roi
        return frame[y : y + h, x : x + w]

    def _preprocess(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return gray

    def _diff_frac(self, a_gray, b_gray):
        diff = cv2.absdiff(a_gray, b_gray)
        _, th = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(th)) / th.size

    def update(self, frame):
        roi_bgr = self._crop(frame)
        cur_gray = self._preprocess(roi_bgr)

        if cur_gray.shape != self.closed_ref_gray.shape:
            raise RuntimeError(
                f"ROI/ref size mismatch: cur={cur_gray.shape}, ref={self.closed_ref_gray.shape}"
            )

        delta_closed_frac = self._diff_frac(cur_gray, self.closed_ref_gray)

        if self.prev_gray is None:
            self.prev_gray = cur_gray
            return 0.0, delta_closed_frac

        motion_frac = self._diff_frac(cur_gray, self.prev_gray)
        self.prev_gray = cur_gray

        return motion_frac, delta_closed_frac
