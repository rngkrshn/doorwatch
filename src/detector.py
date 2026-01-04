# src/detector.py
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

ROI = Tuple[int, int, int, int]


def _crop(frame: np.ndarray, roi: ROI) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@dataclass
class DetectorOutput:
    motion_frac: float
    delta_closed_frac: float


class MotionDetector:
    """
    Computes:
      - motion_frac: difference vs previous frame (useful to detect "something moved")
      - delta_closed_frac: difference vs a closed-door reference frame (the actual door state signal)
    """

    def __init__(
        self,
        roi: ROI,
        diff_threshold: int = 25,
        blur_ksize: int = 11,
        closed_ref_path: str = "data/closed_ref.jpg",
    ):
        self.roi = roi
        self.diff_threshold = diff_threshold
        self.blur_ksize = blur_ksize

        ref = cv2.imread(closed_ref_path)
        if ref is None:
            raise RuntimeError(f"Could not load closed_ref image at: {closed_ref_path}")

        ref_roi = _crop(ref, self.roi)
        ref_gray = _to_gray(ref_roi)

        if self.blur_ksize and self.blur_ksize > 1:
            ref_gray = cv2.GaussianBlur(ref_gray, (self.blur_ksize, self.blur_ksize), 0)

        self.closed_ref_gray = ref_gray
        self.prev_gray: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> DetectorOutput:
        roi_img = _crop(frame, self.roi)
        gray = _to_gray(roi_img)

        if self.blur_ksize and self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # 1) Delta vs CLOSED reference (this is what your state machine should use)
        diff_closed = cv2.absdiff(gray, self.closed_ref_gray)
        _, mask_closed = cv2.threshold(
            diff_closed, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        delta_closed_frac = float(np.count_nonzero(mask_closed)) / mask_closed.size

        # 2) Motion vs previous frame (optional debugging / gating)
        if self.prev_gray is None:
            self.prev_gray = gray
            return DetectorOutput(motion_frac=0.0, delta_closed_frac=delta_closed_frac)

        diff_prev = cv2.absdiff(gray, self.prev_gray)
        _, mask_prev = cv2.threshold(
            diff_prev, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        motion_frac = float(np.count_nonzero(mask_prev)) / mask_prev.size

        self.prev_gray = gray
        return DetectorOutput(motion_frac=motion_frac, delta_closed_frac=delta_closed_frac)
