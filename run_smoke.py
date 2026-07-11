"""One-command smoke test: generate a synthetic sequence, run the tracker,
print REAL measured MOT metrics.

    python run_smoke.py                 # default 6 objects, 200 frames
    python run_smoke.py --frames 300 --objects 8 --seed 7

The reported numbers are computed on a *controlled synthetic sequence* (clearly
labelled). They are genuine outputs of running the tracker — nothing hard-coded.
If ``motmetrics`` is installed, its numbers are printed alongside ours as a
cross-check.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tracker import Sort
from eval import generate_sequence, evaluate_mot


def run(seq, max_age, n_init, iou_threshold):
    """Run the tracker over a sequence; return (hypotheses, fps)."""
    tracker = Sort(max_age=max_age, n_init=n_init, iou_threshold=iou_threshold)
    hyp = {}
    t0 = time.perf_counter()
    for f in range(seq.num_frames):
        dets = seq.dets.get(f, np.empty((0, 5)))
        tracks = tracker.update(dets)
        hyp[f] = tracks
    dt = time.perf_counter() - t0
    fps = seq.num_frames / dt if dt > 0 else float("inf")
    return hyp, fps


def try_motmetrics(gt, hyp, iou_threshold):
    """Optional cross-check against the reference motmetrics library."""
    try:
        # motmetrics 1.4.0 predates numpy 2.0 and calls the removed
        # np.asfarray; shim it so the reference cross-check still runs.
        if not hasattr(np, "asfarray"):
            np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
        import motmetrics as mm
    except Exception:
        return None
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt) | set(hyp))
    for f in frames:
        gt_frame = gt.get(f, [])
        g_ids = [g for g, _ in gt_frame]
        g_boxes = np.array([b for _, b in gt_frame]) if gt_frame else np.empty((0, 4))
        h = np.asarray(hyp.get(f, np.empty((0, 5))))
        h_ids = [int(x[4]) for x in h] if h.size else []
        h_boxes = h[:, :4] if h.size else np.empty((0, 4))
        # motmetrics wants [x, y, w, h] and a distance matrix (1 - IoU).
        def to_xywh(b):
            if b.size == 0:
                return b
            out = b.copy()
            out[:, 2] = b[:, 2] - b[:, 0]
            out[:, 3] = b[:, 3] - b[:, 1]
            return out
        dist = mm.distances.iou_matrix(to_xywh(g_boxes), to_xywh(h_boxes),
                                       max_iou=1 - iou_threshold)
        acc.update(g_ids, h_ids, dist)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches"],
                         name="ref")
    return summary


def main():
    ap = argparse.ArgumentParser(description="MOT tracker smoke test")
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--miss-rate", type=float, default=0.10)
    ap.add_argument("--fp-rate", type=float, default=0.20)
    ap.add_argument("--noise-std", type=float, default=3.0)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--iou-threshold", type=float, default=0.3,
                    help="association IoU gate (tracker)")
    ap.add_argument("--eval-iou", type=float, default=0.5,
                    help="IoU threshold for metric matching")
    args = ap.parse_args()

    print("=" * 66)
    print(" MOT smoke test  —  CONTROLLED SYNTHETIC SEQUENCE (not MOT16)")
    print("=" * 66)
    print(f" objects={args.objects}  frames={args.frames}  seed={args.seed}")
    print(f" detector sim: miss_rate={args.miss_rate}  fp_rate={args.fp_rate}"
          f"  noise_std={args.noise_std}px")
    print(f" tracker: max_age={args.max_age}  n_init={args.n_init}"
          f"  assoc_iou={args.iou_threshold}  eval_iou={args.eval_iou}")
    print("-" * 66)

    seq = generate_sequence(
        num_objects=args.objects, num_frames=args.frames, seed=args.seed,
        miss_rate=args.miss_rate, fp_rate=args.fp_rate, noise_std=args.noise_std)

    # Report input detection quality for context (raw detector vs GT).
    det_metrics = evaluate_mot(
        seq.gt, {f: _dets_as_hyp(seq.dets[f]) for f in seq.dets},
        iou_threshold=args.eval_iou)

    hyp, fps = run(seq, args.max_age, args.n_init, args.iou_threshold)
    m = evaluate_mot(seq.gt, hyp, iou_threshold=args.eval_iou)

    print(" RAW DETECTIONS vs GT (before tracking):")
    print(f"   FP={det_metrics.num_fp}  FN={det_metrics.num_misses}"
          f"  (shows the tracker has real noise to fight)")
    print("-" * 66)
    print(" TRACKER RESULT (measured):")
    for k, v in m.as_dict().items():
        if isinstance(v, float):
            print(f"   {k:16s}: {v:.4f}")
        else:
            print(f"   {k:16s}: {v}")
    print(f"   {'throughput':16s}: {fps:.0f} FPS (assoc+KF only, CPU)")
    print("-" * 66)

    ref = try_motmetrics(seq.gt, hyp, args.eval_iou)
    if ref is not None:
        print(" CROSS-CHECK vs motmetrics library:")
        print(ref.to_string())
    else:
        print(" (motmetrics not installed — skipping reference cross-check)")
    print("=" * 66)


def _dets_as_hyp(dets):
    """Treat raw detections as pseudo-tracks (id=index) for a quality readout."""
    if dets is None or len(dets) == 0:
        return np.empty((0, 5))
    out = dets.copy()
    out[:, 4] = np.arange(len(dets))  # fake unique ids
    return out


if __name__ == "__main__":
    main()
