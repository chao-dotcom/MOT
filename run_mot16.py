"""Run the tracker on a real MOT16/MOT17 sequence and (optionally) score it.

This is the *scale-up* entry point. It does NOT download any data — see the
README for the (free) MOTChallenge download and directory layout. Point it at
a sequence folder that already contains ``det/det.txt`` (and ``gt/gt.txt`` for
scoring on the train split).

Examples
--------
Score against ground truth (train split):
    python run_mot16.py --seq path/to/MOT16/train/MOT16-02

Produce a submission file for the test split (no gt):
    python run_mot16.py --seq path/to/MOT16/test/MOT16-01 --out results/MOT16-01.txt

Layout expected (standard MOTChallenge):
    <seq>/det/det.txt
    <seq>/gt/gt.txt          (train only)
    <seq>/seqinfo.ini        (optional; not required)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from tracker import Sort
from tracker.detectors import FileDetector
from eval.mot_io import read_det, read_gt, write_results
from eval.metrics import evaluate_mot


def main():
    ap = argparse.ArgumentParser(description="Run tracker on a MOT16/17 sequence")
    ap.add_argument("--seq", required=True, help="path to a sequence folder")
    ap.add_argument("--out", default=None, help="write MOT-format results here")
    ap.add_argument("--det-score", type=float, default=0.3,
                    help="detection confidence threshold")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--eval-iou", type=float, default=0.5)
    args = ap.parse_args()

    det_path = os.path.join(args.seq, "det", "det.txt")
    gt_path = os.path.join(args.seq, "gt", "gt.txt")
    if not os.path.isfile(det_path):
        raise SystemExit(f"detections not found: {det_path}\n"
                         "See README for the MOTChallenge download + layout.")

    det_by_frame = read_det(det_path, score_thresh=args.det_score)
    detector = FileDetector(det_by_frame)
    frames = sorted(det_by_frame) or [1]
    last = max(frames)

    tracker = Sort(max_age=args.max_age, n_init=args.n_init,
                   iou_threshold=args.iou_threshold)
    hyp = {}
    t0 = time.perf_counter()
    for f in range(1, last + 1):        # MOT frames are 1-indexed and contiguous
        hyp[f] = tracker.update(detector(f))
    dt = time.perf_counter() - t0
    print(f"Processed {last} frames in {dt:.2f}s "
          f"({last / dt:.0f} FPS, tracking only).")

    if args.out:
        write_results(args.out, hyp)
        print(f"Wrote results -> {args.out}")

    if os.path.isfile(gt_path):
        gt = read_gt(gt_path, only_pedestrian=True)
        m = evaluate_mot(gt, hyp, iou_threshold=args.eval_iou)
        print("-" * 50)
        print(f" REAL MOT metrics for {os.path.basename(args.seq.rstrip('/'))}:")
        for k, v in m.as_dict().items():
            print(f"   {k:16s}: {v:.4f}" if isinstance(v, float)
                  else f"   {k:16s}: {v}")
    else:
        print("(no gt/gt.txt — this is a test sequence; "
              "submit the results file to MOTChallenge to get scored.)")


if __name__ == "__main__":
    main()
