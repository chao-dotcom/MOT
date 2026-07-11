"""Lightweight self-tests (no pytest required):  python tests/test_basic.py

Covers the load-bearing pieces: IoU correctness, the Kalman filter tracking a
constant-velocity target, perfect-detection tracking (should be near-perfect),
and the metrics matching an exactly-correct hypothesis.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import Sort
from tracker.association import iou_batch, associate
from tracker.kalman import KalmanBoxFilter
from eval import generate_sequence, evaluate_mot


def test_iou():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10],       # identical -> 1.0
                  [5, 0, 15, 10],       # half overlap -> 1/3
                  [20, 20, 30, 30]],    # disjoint -> 0
                 dtype=float)
    iou = iou_batch(a, b)
    assert abs(iou[0, 0] - 1.0) < 1e-9
    assert abs(iou[0, 1] - (1 / 3)) < 1e-6
    assert iou[0, 2] == 0.0
    print("ok  iou_batch")


def test_associate_basic():
    dets = np.array([[0, 0, 10, 10, 1.0], [100, 100, 110, 110, 1.0]])
    trks = np.array([[1, 1, 11, 11], [100, 100, 110, 110]])
    matches, ud, ut = associate(dets, trks, 0.3)
    assert len(matches) == 2 and len(ud) == 0 and len(ut) == 0
    print("ok  associate")


def test_kalman_constant_velocity():
    # Object moving right at 5 px/frame; KF prediction should track it.
    box = np.array([100.0, 100.0, 140.0, 140.0])
    kf = KalmanBoxFilter(box)
    for step in range(1, 20):
        box = box + np.array([5, 0, 5, 0])
        kf.predict()
        kf.update(box)
    pred = kf.predict()  # one step ahead
    expected_cx = (box[0] + box[2]) / 2 + 5
    got_cx = (pred[0] + pred[2]) / 2
    assert abs(got_cx - expected_cx) < 3.0, (got_cx, expected_cx)
    print("ok  kalman constant-velocity")


def test_perfect_detections():
    # With clean detections and no clutter, MOTA/IDF1 should be very high.
    seq = generate_sequence(num_objects=4, num_frames=100,
                            miss_rate=0.0, fp_rate=0.0, noise_std=0.0, seed=3)
    t = Sort()
    hyp = {f: t.update(seq.dets[f]) for f in range(seq.num_frames)}
    m = evaluate_mot(seq.gt, hyp, 0.5)
    assert m.mota > 0.95, m.mota
    assert m.idf1 > 0.95, m.idf1
    assert m.num_switches == 0, m.num_switches
    print(f"ok  perfect-detections MOTA={m.mota:.3f} IDF1={m.idf1:.3f}")


def test_metrics_identity():
    # Feeding GT back as the hypothesis must give MOTA==IDF1==1, zero errors.
    seq = generate_sequence(num_objects=3, num_frames=50, seed=5)
    hyp = {}
    for f in range(seq.num_frames):
        rows = [np.concatenate([b, [oid]]) for oid, b in seq.gt[f]]
        hyp[f] = np.stack(rows) if rows else np.empty((0, 5))
    m = evaluate_mot(seq.gt, hyp, 0.5)
    assert abs(m.mota - 1.0) < 1e-9 and abs(m.idf1 - 1.0) < 1e-9
    assert m.num_fp == 0 and m.num_misses == 0 and m.num_switches == 0
    print("ok  metrics identity (perfect hypothesis)")


if __name__ == "__main__":
    test_iou()
    test_associate_basic()
    test_kalman_constant_velocity()
    test_perfect_detections()
    test_metrics_identity()
    print("\nALL TESTS PASSED")
