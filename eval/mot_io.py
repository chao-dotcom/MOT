"""Read/write MOTChallenge (MOT16/MOT17) format files.

MOT det/gt line format (comma-separated):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

For detections ``id`` is -1 and ``conf`` is the score. For ground truth ``conf``
is a "consider" flag (1/0) and columns 8-10 hold class/visibility. We keep this
minimal and dependency-free (plain file parsing).
"""
from __future__ import annotations

import os

import numpy as np


def read_det(path: str, score_thresh: float = 0.0) -> dict:
    """Read a MOT det.txt into {frame_idx: (N,5) [x1,y1,x2,y2,score]}."""
    out: dict = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame = int(float(p[0]))
            left, top, w, h = (float(p[2]), float(p[3]),
                               float(p[4]), float(p[5]))
            score = float(p[6]) if len(p) > 6 else 1.0
            if score < score_thresh:
                continue
            box = [left, top, left + w, top + h, score]
            out.setdefault(frame, []).append(box)
    return {f: np.array(v, dtype=np.float64) for f, v in out.items()}


def read_gt(path: str, only_pedestrian: bool = True) -> dict:
    """Read a MOT gt.txt into {frame_idx: [(obj_id, [x1,y1,x2,y2]), ...]}.

    When ``only_pedestrian`` is set, keep rows with the 'consider' flag on and
    class id 1 (pedestrian), matching the MOT16 evaluation convention.
    """
    out: dict = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame = int(float(p[0]))
            obj_id = int(float(p[1]))
            left, top, w, h = (float(p[2]), float(p[3]),
                               float(p[4]), float(p[5]))
            consider = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else 1
            if only_pedestrian and (consider < 1 or cls != 1):
                continue
            out.setdefault(frame, []).append(
                (obj_id, [left, top, left + w, top + h]))
    return out


def write_results(path: str, hyp: dict) -> None:
    """Write tracker outputs to a MOT-format results file.

    Args:
        path: destination file.
        hyp:  {frame_idx: (N,5) array of [x1,y1,x2,y2,track_id]}.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for frame in sorted(hyp):
            arr = np.asarray(hyp[frame])
            for row in arr:
                x1, y1, x2, y2, tid = row
                w, h = x2 - x1, y2 - y1
                fh.write(f"{frame},{int(tid)},{x1:.2f},{y1:.2f},"
                         f"{w:.2f},{h:.2f},1,-1,-1,-1\n")
