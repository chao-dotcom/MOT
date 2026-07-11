"""SORT: Simple Online and Realtime Tracking (tracking-by-detection).

Pipeline per frame:
    1. Predict every existing track forward with its Kalman filter.
    2. Associate incoming detections to predicted boxes by IoU (Hungarian/greedy).
    3. Update matched tracks; spawn tracks for unmatched detections;
       coast/kill unmatched tracks via the lifecycle rules in ``Track``.
    4. Emit the currently reportable tracks.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
This is an original numpy re-implementation for a portfolio, not a fork.
"""
from __future__ import annotations

import numpy as np

from .association import associate
from .track import Track, TrackState


class Sort:
    """Multi-object tracker managing a set of :class:`Track` objects."""

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 iou_threshold: float = 0.3):
        """
        Args:
            max_age:       frames a track may coast unmatched before deletion.
            n_init:        hits required before a track is CONFIRMED / reported.
            iou_threshold: IoU gate for detection<->track association.
        """
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.iou_threshold = float(iou_threshold)
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: np.ndarray | None = None) -> np.ndarray:
        """Advance the tracker by one frame.

        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, score], or None/empty.

        Returns:
            (M, 5) array of reported tracks as [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=np.float64)
        else:
            detections = np.asarray(detections, dtype=np.float64)

        # 1. Predict existing tracks. Drop any that produce invalid (NaN) boxes.
        predicted = np.zeros((len(self.tracks), 4), dtype=np.float64)
        valid = []
        for i, trk in enumerate(self.tracks):
            pos = trk.predict()
            predicted[i] = pos
            if not np.any(np.isnan(pos)):
                valid.append(i)
        if len(valid) != len(self.tracks):
            predicted = predicted[valid]
            self.tracks = [self.tracks[i] for i in valid]

        # 2. Associate detections to predicted track boxes.
        matches, unmatched_dets, unmatched_trks = associate(
            detections, predicted, self.iou_threshold)

        # 3a. Update matched tracks with their detection.
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx, :4])

        # 3b. Mark unmatched tracks as missed (coast or die).
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()

        # 3c. Birth: create a new track for every unmatched detection.
        for det_idx in unmatched_dets:
            self.tracks.append(
                Track(detections[det_idx, :4], self._next_id,
                      n_init=self.n_init, max_age=self.max_age))
            self._next_id += 1

        # 4. Collect outputs and reap deleted tracks.
        outputs = []
        surviving = []
        for trk in self.tracks:
            if not trk.is_deleted():
                surviving.append(trk)
            # Report a track that was updated this frame and is either confirmed
            # or still within the warm-up window (so early frames aren't empty).
            reportable = (trk.time_since_update < 1) and (
                trk.is_confirmed() or self.frame_count <= self.n_init)
            if reportable and not trk.is_deleted():
                outputs.append(np.concatenate([trk.bbox, [trk.track_id]]))
        self.tracks = surviving

        if outputs:
            return np.stack(outputs, axis=0)
        return np.empty((0, 5), dtype=np.float64)
