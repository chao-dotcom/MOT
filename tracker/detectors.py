"""Detector adapters: feed the tracker with real detections.

Two sources are supported so the tracker is decoupled from any specific
detector and the pipeline runs with or without heavy deps:

1. ``FileDetector`` — reads detections from a MOT-format ``det.txt`` (the format
   used by MOT16/17 and MOTChallenge). No model download required.

2. ``YoloDetector`` — optional wrapper around Ultralytics YOLO. Only imported
   when explicitly constructed, so ``import tracker`` never pulls in torch.
   Guarded so a missing install fails with a clear message instead of a crash.

Both expose the same interface::

    dets = detector(frame_index)          # or detector(image) for YOLO
    # -> (N, 5) array of [x1, y1, x2, y2, score]
"""
from __future__ import annotations

import numpy as np


class FileDetector:
    """Serve pre-computed detections keyed by 1-based frame index.

    Args:
        det_by_frame: dict frame_idx -> (N,5) array of [x1,y1,x2,y2,score].
        score_thresh: drop detections below this confidence.
    """

    def __init__(self, det_by_frame: dict, score_thresh: float = 0.0):
        self.det_by_frame = det_by_frame
        self.score_thresh = float(score_thresh)

    def __call__(self, frame_index: int) -> np.ndarray:
        dets = self.det_by_frame.get(frame_index, np.empty((0, 5)))
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        if self.score_thresh > 0 and len(dets):
            dets = dets[dets[:, 4] >= self.score_thresh]
        return dets


class YoloDetector:  # pragma: no cover - requires optional ultralytics+torch
    """Optional Ultralytics YOLO adapter (person class by default).

    Construct only when you actually want to run a model:

        det = YoloDetector(weights="yolov8n.pt", classes=[0])
        boxes = det(image_bgr)   # numpy HxWx3

    Kept import-light: ``ultralytics`` is imported lazily in ``__init__`` so the
    rest of the project runs in a pure-numpy environment.
    """

    def __init__(self, weights: str = "yolov8n.pt",
                 conf: float = 0.3, classes=None):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # clear, actionable error
            raise ImportError(
                "YoloDetector requires 'ultralytics'. Install with "
                "`pip install ultralytics` (pulls torch). For a dependency-free "
                "run, use FileDetector with a MOT-format det.txt instead."
            ) from exc
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.classes = classes  # e.g. [0] for the COCO 'person' class

    def __call__(self, image) -> np.ndarray:
        res = self.model.predict(image, conf=self.conf, classes=self.classes,
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 5), dtype=np.float64)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf]).astype(np.float64)
