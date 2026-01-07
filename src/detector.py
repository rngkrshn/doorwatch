# src/detector.py
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class MotionDetector:
    """
    Computes:
      - motion_frac: fraction of ROI pixels that changed vs previous frame
      - delta_closed: fraction of ROI pixels that differ from a "closed reference" frame

    Supports self-calibration:
      - If closed_ref_gray is not provided, it will wait for the scene to be stable
        for a short window and then capture a closed reference automatically.
    """

    def __init__(
        self,
        roi: Tuple[int, int, int, int],
        diff_threshold: int = 25,
        blur_ksize: int = 11,
        closed_ref_gray: Optional[np.ndarray] = None,
        stable_motion_threshold: float = 0.002,   # what counts as "stable" for calibration
        stable_required_frames: int = 20,         # how many stable frames before we capture ref
        max_calibration_seconds: float = 20.0,    # safety timeout
    ):
        self.roi = roi
        self.diff_threshold = int(diff_threshold)
        self.blur_ksize = int(blur_ksize)

        self._prev_gray: Optional[np.ndarray] = None
        self._closed_ref_gray: Optional[np.ndarray] = closed_ref_gray

        # calibration state
        self._stable_motion_threshold = float(stable_motion_threshold)
        self._stable_required_frames = int(stable_required_frames)
        self._stable_count = 0
        self._calib_start = time.time()
        self._max_calib_seconds = float(max_calibration_seconds)

    @property
    def calibrated(self) -> bool:
        return self._closed_ref_gray is not None

    def set_closed_ref(self, closed_ref_gray: np.ndarray) -> None:
        self._closed_ref_gray = closed_ref_gray.copy()

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return frame[y : y + h, x : x + w]

    def _to_gray_blur(self, frame: np.ndarray) -> np.ndarray:
        roi = self._crop(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        return gray

    def _diff_frac(self, a: np.ndarray, b: np.ndarray) -> float:
        # fraction of pixels whose abs diff exceeds diff_threshold
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        changed = float(np.count_nonzero(mask))
        total = float(mask.size)
        return changed / total if total > 0 else 0.0

    def update(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Returns: (motion_frac, delta_closed)
        - motion_frac uses prev frame
        - delta_closed uses the closed reference frame (0.0 until calibrated)
        """
        gray = self._to_gray_blur(frame)

        # motion vs previous frame
        if self._prev_gray is None:
            motion = 0.0
        else:
            motion = self._diff_frac(gray, self._prev_gray)

        # self-calibration: wait for stable frames, then capture reference
        if not self.calibrated:
            if time.time() - self._calib_start > self._max_calib_seconds:
                # If we can't get stable frames, still pick the current frame as ref
                self._closed_ref_gray = gray.copy()
            else:
                if motion <= self._stable_motion_threshold:
                    self._stable_count += 1
                else:
                    self._stable_count = 0

                if self._stable_count >= self._stable_required_frames:
                    self._closed_ref_gray = gray.copy()

        # delta vs closed reference
        if self.calibrated:
            delta_closed = self._diff_frac(gray, self._closed_ref_gray)  # type: ignore[arg-type]
        else:
            delta_closed = 0.0

        self._prev_gray = gray
        return motion, delta_closed
