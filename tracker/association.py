"""Data association: IoU cost + optimal (Hungarian) or greedy matching.

This module is detector/tracker-agnostic — it just matches two sets of boxes.
It uses scipy's ``linear_sum_assignment`` (Hungarian algorithm) when available
and falls back to a greedy highest-IoU-first matcher otherwise, so the project
runs even in a bare numpy environment.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised only without scipy
    _HAS_SCIPY = False


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2].
        boxes_b: (M, 4) array of [x1, y1, x2, y2].
    Returns:
        (N, M) IoU matrix. Empty inputs yield a correctly-shaped empty/zero array.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    a = boxes_a[:, None, :]   # (N,1,4)
    b = boxes_b[None, :, :]   # (1,M,4)

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(xx2 - xx1, 0.0, None)
    inter_h = np.clip(yy2 - yy1, 0.0, None)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def _greedy_match(iou: np.ndarray, iou_threshold: float):
    """Greedy fallback: repeatedly take the globally highest IoU pair."""
    matches = []
    iou = iou.copy()
    while True:
        idx = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[idx] < iou_threshold or iou[idx] <= 0:
            break
        matches.append((int(idx[0]), int(idx[1])))
        iou[idx[0], :] = -1.0   # remove this track row
        iou[:, idx[1]] = -1.0   # remove this detection column
    return matches


def associate(detections: np.ndarray,
              trackers: np.ndarray,
              iou_threshold: float = 0.3):
    """Associate detections to predicted tracker boxes via IoU.

    Args:
        detections: (N, 4+) detection boxes (only first 4 cols used).
        trackers:   (M, 4)  predicted tracker boxes.
        iou_threshold: minimum IoU to accept a match (gating).

    Returns:
        matches:            list of (det_idx, trk_idx) pairs.
        unmatched_dets:     np.ndarray of detection indices with no match.
        unmatched_trks:     np.ndarray of tracker indices with no match.
    """
    n_det = detections.shape[0]
    n_trk = trackers.shape[0]

    if n_trk == 0:
        return [], np.arange(n_det), np.empty(0, dtype=int)
    if n_det == 0:
        return [], np.empty(0, dtype=int), np.arange(n_trk)

    iou = iou_batch(detections[:, :4], trackers[:, :4])  # (N_det, M_trk)

    if _HAS_SCIPY:
        # Hungarian maximises IoU by minimising -IoU.
        det_ind, trk_ind = linear_sum_assignment(-iou)
        candidate = list(zip(det_ind.tolist(), trk_ind.tolist()))
        matches = [(d, t) for d, t in candidate if iou[d, t] >= iou_threshold]
    else:  # pragma: no cover
        matches = _greedy_match(iou, iou_threshold)

    matched_dets = {d for d, _ in matches}
    matched_trks = {t for _, t in matches}
    unmatched_dets = np.array([d for d in range(n_det) if d not in matched_dets],
                              dtype=int)
    unmatched_trks = np.array([t for t in range(n_trk) if t not in matched_trks],
                              dtype=int)
    return matches, unmatched_dets, unmatched_trks
