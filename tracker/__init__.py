"""SORT-style multi-object tracker (Kalman filter + IoU data association).

Public API:
    from tracker import Sort
    tracker = Sort(...)
    tracks = tracker.update(detections)  # detections: (N,5) [x1,y1,x2,y2,score]
"""
from .sort import Sort
from .track import Track, TrackState

__all__ = ["Sort", "Track", "TrackState"]
