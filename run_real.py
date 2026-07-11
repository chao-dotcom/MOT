"""Turnkey REAL MOT17 benchmark: download labels -> track -> score.

This runs the from-scratch SORT tracker (Kalman + Hungarian association) on
**real MOT17 train sequences** using the **public detections** shipped with the
benchmark, and scores the output against ground truth with the reference
``motmetrics`` library (CLEAR-MOT + IDF1 + MT/ML).

Efficiency note: tracking-by-detection only needs the small TEXT label files
(``det/det.txt`` + ``gt/gt.txt``) — NOT the multi-GB image frames. This script
downloads only those label files (a few MB total) from a GitHub/HF mirror of the
MOTChallenge labels. It never touches the images.

Usage
-----
    python run_real.py                 # download (if needed) + run 4 seqs + table
    python run_real.py --det-score 0.5 # sweep the detection confidence gate
    python run_real.py --no-download   # use only what is already on disk
    python run_real.py --seqs MOT17-09-FRCNN   # subset

Data layout produced (standard MOTChallenge):
    data/MOT17/train/<SEQ>/det/det.txt
    data/MOT17/train/<SEQ>/gt/gt.txt

Honesty: these are REAL public-detection numbers. SORT on MOT17 public dets
scores ~MOTA 40-60 (missed detections dominate the error) — much lower than a
clean synthetic sequence, and that is expected and more credible.
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

import numpy as np

# motmetrics 1.4.0 predates numpy 2.0 and calls the removed np.asfarray; shim it
# so the reference evaluator runs on modern numpy. Must precede the mm import.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from tracker import Sort
from tracker.detectors import FileDetector
from eval.mot_io import read_det, read_gt

# Public-detection train sequences (FRCNN detector) — the standard SORT eval set.
DEFAULT_SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
]

# Labels-only mirror of the MOTChallenge MOT17 train split (det.txt + gt.txt).
# We fetch ONLY the two text files per sequence, never the img1/ frames.
MIRROR = "https://huggingface.co/datasets/Lekim89/MOT17/resolve/main/train"


def _download(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "mot-tracker/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as fh:
        fh.write(r.read())


def ensure_labels(seq: str, data_dir: str, download: bool = True) -> str:
    """Return the sequence folder, downloading det/gt labels if missing."""
    seq_dir = os.path.join(data_dir, "MOT17", "train", seq)
    det = os.path.join(seq_dir, "det", "det.txt")
    gt = os.path.join(seq_dir, "gt", "gt.txt")
    for rel, path in (("det/det.txt", det), ("gt/gt.txt", gt)):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        if not download:
            raise SystemExit(
                f"missing {path} and --no-download set. Fetch it from "
                f"{MIRROR}/{seq}/{rel} or MOTChallenge (train labels).")
        url = f"{MIRROR}/{seq}/{rel}"
        print(f"  downloading {seq}/{rel} ...", flush=True)
        _download(url, path)
    return seq_dir


def _ltwh(b):
    """[x1,y1,x2,y2] -> [left,top,w,h] as motmetrics expects."""
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def run_sequence(seq_dir: str, det_score: float, max_age: int, n_init: int,
                 iou_threshold: float, eval_iou: float):
    """Track one sequence and return (motmetrics accumulator, fps)."""
    import motmetrics as mm

    det = read_det(os.path.join(seq_dir, "det", "det.txt"),
                   score_thresh=det_score)
    gt = read_gt(os.path.join(seq_dir, "gt", "gt.txt"), only_pedestrian=True)
    detector = FileDetector(det)
    last = max(max(det, default=1), max(gt, default=1))

    tracker = Sort(max_age=max_age, n_init=n_init, iou_threshold=iou_threshold)
    acc = mm.MOTAccumulator(auto_id=True)

    t0 = time.perf_counter()
    for f in range(1, last + 1):
        out = tracker.update(detector(f))
        g = gt.get(f, [])
        gt_ids = [i for i, _ in g]
        gt_boxes = [_ltwh(b) for _, b in g]
        hyp_ids = [int(r[4]) for r in out]
        hyp_boxes = [_ltwh(r[:4]) for r in out]
        dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes,
                                       max_iou=1.0 - eval_iou)
        acc.update(gt_ids, hyp_ids, dist)
    fps = last / (time.perf_counter() - t0)
    return acc, fps, last


def main():
    ap = argparse.ArgumentParser(description="Real MOT17 benchmark (public dets)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seqs", nargs="+", default=DEFAULT_SEQS)
    ap.add_argument("--det-score", type=float, default=0.3,
                    help="detection confidence gate (MOT17 dets are 0..1)")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--iou-threshold", type=float, default=0.3,
                    help="IoU gate for detection<->track association")
    ap.add_argument("--eval-iou", type=float, default=0.5,
                    help="IoU required to call a gt/hyp pair a match")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    import motmetrics as mm

    print("=" * 78)
    print(" REAL MOT17 benchmark  --  from-scratch SORT on PUBLIC detections")
    print("=" * 78)
    print(f" seqs={args.seqs}")
    print(f" tracker: max_age={args.max_age} n_init={args.n_init} "
          f"assoc_iou={args.iou_threshold} det_gate={args.det_score} "
          f"eval_iou={args.eval_iou}")
    print("-" * 78)

    accs, names, fps_all = [], [], []
    for seq in args.seqs:
        seq_dir = ensure_labels(seq, args.data_dir, download=not args.no_download)
        acc, fps, nframes = run_sequence(
            seq_dir, args.det_score, args.max_age, args.n_init,
            args.iou_threshold, args.eval_iou)
        accs.append(acc)
        names.append(seq)
        fps_all.append(fps)
        print(f"  {seq}: {nframes} frames @ {fps:.0f} FPS (tracking only)")

    metrics = ["mota", "motp", "idf1", "idp", "idr", "num_switches",
               "mostly_tracked", "mostly_lost", "num_false_positives",
               "num_misses", "num_fragmentations", "num_objects"]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=metrics,
                              generate_overall=True)

    # motp comes back as mean distance (1 - IoU); show IoU too for readability.
    summary = summary.copy()
    summary["motp_iou"] = 1.0 - summary["motp"]

    print("-" * 78)
    rename = {
        "mota": "MOTA", "motp_iou": "MOTP", "idf1": "IDF1",
        "num_switches": "IDsw", "mostly_tracked": "MT", "mostly_lost": "ML",
        "num_false_positives": "FP", "num_misses": "FN",
        "num_fragmentations": "Frag", "num_objects": "GT",
    }
    show = summary[list(rename)].rename(columns=rename)
    fmt = {"MOTA": "{:.3f}", "MOTP": "{:.3f}", "IDF1": "{:.3f}"}
    with_fmt = show.copy()
    for c, f in fmt.items():
        with_fmt[c] = with_fmt[c].map(lambda v: f.format(v))
    print(with_fmt.to_string())
    print("-" * 78)
    o = summary.loc["OVERALL"]
    print(f" OVERALL: MOTA={o.mota:.3f}  IDF1={o.idf1:.3f}  "
          f"MOTP(IoU)={o.motp_iou:.3f}  IDsw={int(o.num_switches)}  "
          f"MT={int(o.mostly_tracked)}  ML={int(o.mostly_lost)}  "
          f"FP={int(o.num_false_positives)}  FN={int(o.num_misses)}")
    print(f" mean tracking throughput: {np.mean(fps_all):.0f} FPS (CPU, "
          f"assoc+Kalman only)")
    print("=" * 78)
    print("Note: scored with the reference `motmetrics` library. Numbers use the"
          "\nraw public detections without the official MOTChallenge distractor/"
          "\nignore-region preprocessing, so they are a slight lower bound vs the"
          "\nleaderboard; SORT on public dets lands ~MOTA 40-60 as expected.")


if __name__ == "__main__":
    main()
