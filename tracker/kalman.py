"""Constant-velocity Kalman filter for a single bounding-box track.

State vector (7-D), following the classic SORT parameterisation:

    x = [u, v, s, r, u', v', s']^T

where
    u, v : centre of the box (pixels)
    s    : scale = area = w * h
    r    : aspect ratio = w / h  (assumed ~constant, so no velocity term)
    u',v': centre velocities
    s'   : scale (area) velocity

Measurement vector (4-D): z = [u, v, s, r]  (an observed bounding box).

The model is deliberately simple (linear, constant velocity). It is the
standard, well-understood baseline used by SORT and is enough to demonstrate
the state-estimation core: predict / update, process & measurement noise,
and covariance propagation.

Implemented from scratch with numpy so the filter math is explicit and
reviewable — no filterpy / external KF dependency.
"""
from __future__ import annotations

import numpy as np


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] -> measurement [u, v, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h                      # scale (area)
    r = w / float(h) if h > 0 else 0.0
    return np.array([u, v, s, r], dtype=np.float64)


def _x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert state [u, v, s, r, ...] -> [x1, y1, x2, y2].

    Recovers w, h from area s and aspect ratio r:
        w = sqrt(s * r),  h = s / w
    """
    s = max(float(x[2]), 1e-6)
    r = max(float(x[3]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    u, v = float(x[0]), float(x[1])
    return np.array([u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0],
                    dtype=np.float64)


class KalmanBoxFilter:
    """A linear Kalman filter tracking one bounding box under constant velocity."""

    def __init__(self, bbox: np.ndarray):
        ndim, dt = 4, 1.0

        # State-transition matrix F (7x7): constant velocity for u, v, s.
        self.F = np.eye(7, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 4] = dt

        # Measurement matrix H (4x7): we observe u, v, s, r directly.
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise R. Aspect ratio & centre are fairly reliable;
        # scale is noisier. Values follow SORT's tuned defaults.
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Initial covariance P. Large uncertainty on the unobserved velocities.
        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        # Process noise Q. Small drift on positions, smaller on velocities.
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        # Initialise state from the first detection; velocities start at 0.
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4, 0] = _bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        """Advance the state one step and return the predicted bbox.

        Guards against negative predicted area (a physically invalid box),
        which can otherwise happen when s' drives s below zero during coasting.
        """
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _x_to_bbox(self.x[:, 0])

    def update(self, bbox: np.ndarray) -> None:
        """Correct the state with an observed detection [x1,y1,x2,y2]."""
        z = _bbox_to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(7, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        """Current best estimate as [x1, y1, x2, y2]."""
        return _x_to_bbox(self.x[:, 0])
