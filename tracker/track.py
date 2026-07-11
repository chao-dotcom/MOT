"""Per-object track with lifecycle management (birth / confirm / coast / death).

A ``Track`` wraps one Kalman filter and the bookkeeping needed to decide when
a hypothesis is trustworthy enough to report (``CONFIRMED``) and when it should
be dropped after too many missed frames (coasting -> ``DELETED``).
"""
from __future__ import annotations

import enum

import numpy as np

from .kalman import KalmanBoxFilter


class TrackState(enum.IntEnum):
    """Lifecycle states of a track."""
    TENTATIVE = 0   # just born, not yet confirmed
    CONFIRMED = 1   # matched enough times to be reported
    DELETED = 2     # scheduled for removal


class Track:
    """A single tracked object backed by a constant-velocity Kalman filter."""

    def __init__(self, bbox: np.ndarray, track_id: int,
                 n_init: int = 3, max_age: int = 30):
        """
        Args:
            bbox:     initial detection [x1, y1, x2, y2].
            track_id: unique integer ID for this track.
            n_init:   consecutive hits required to move TENTATIVE -> CONFIRMED.
            max_age:  frames a track may coast (unmatched) before deletion.
        """
        self.kf = KalmanBoxFilter(bbox)
        self.track_id = int(track_id)
        self.n_init = int(n_init)
        self.max_age = int(max_age)

        self.state = TrackState.TENTATIVE
        self.hits = 1              # total matched updates
        self.hit_streak = 1        # consecutive matched updates
        self.time_since_update = 0  # frames since last match (coast counter)
        self.age = 1               # total frames alive

    # -- prediction -------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Propagate the KF one frame; update coasting counters."""
        bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0     # a gap breaks the confirmation streak
        self.time_since_update += 1
        return bbox

    # -- correction -------------------------------------------------------
    def update(self, bbox: np.ndarray) -> None:
        """Correct the KF with a matched detection and refresh counters."""
        self.kf.update(bbox)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Called when a track was not matched this frame; may delete it."""
        if self.state == TrackState.TENTATIVE:
            # A tentative track that misses before confirming is discarded.
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    # -- queries ----------------------------------------------------------
    @property
    def bbox(self) -> np.ndarray:
        return self.kf.bbox

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
