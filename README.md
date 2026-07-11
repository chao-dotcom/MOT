# mot_tracker — SORT-style Multi-Object Tracker (Kalman + data association)

A compact, from-scratch **tracking-by-detection** system in Python/numpy:
a per-track **constant-velocity Kalman filter**, **IoU data association** via the
**Hungarian algorithm** (with a greedy fallback), full **track lifecycle**
(birth → confirm → coast → death), and a **from-scratch MOT metrics** suite
(MOTA / MOTP / IDF1 / ID-switches) validated against the reference
`motmetrics` library.

It produces **genuine, measured numbers today** on both a controlled synthetic
benchmark **and real MOT17 train sequences with public detections** (one command,
labels auto-downloaded — see `run_real.py`).

> Author: Chao Hsuan Ho. Built as an edge-AI / perception portfolio artifact.
> The Kalman/state-estimation core reflects hands-on estimation experience.

---

## Quick start (no downloads, ~1 second)

```bash
pip install numpy scipy          # scipy optional (greedy fallback exists)
python run_smoke.py              # prints REAL measured MOTA / IDF1 / ID-switch
python tests/test_basic.py       # self-tests (IoU, KF, metrics identity)
```

`run_smoke.py` builds a controlled synthetic sequence with **known ground
truth** (objects moving at constant velocity, bouncing off edges, crossing each
other), simulates a **noisy detector** (localisation noise + missed detections +
false positives), runs the tracker, and reports measured metrics. If
`motmetrics` is installed it prints the library's numbers alongside ours as a
cross-check.

See [`RESULTS.md`](RESULTS.md) for the actual numbers from a real run.

---

## What's inside

```
tracker/
  kalman.py        Constant-velocity Kalman filter (7-D state) for one box
  association.py   Vectorised IoU + Hungarian (scipy) / greedy matching + gating
  track.py         Track lifecycle: TENTATIVE → CONFIRMED → coast → DELETED
  sort.py          Sort: the online tracker tying it together
  detectors.py     FileDetector (MOT det.txt) + optional YoloDetector (ultralytics)
eval/
  synthetic.py     Controlled sequence generator with known GT + simulated detector
  metrics.py       From-scratch CLEAR MOT (MOTA/MOTP, FP/FN/IDSW) + IDF1
  mot_io.py        Read/write MOTChallenge (MOT16/17) det/gt/results files
run_smoke.py       One command → measured metrics on the synthetic sequence
run_mot16.py       Real-data entry point (documented below)
tests/test_basic.py  Dependency-free self-tests
```

### Algorithm notes
- **Kalman filter** — SORT's 7-D parameterisation `[u, v, s, r, u', v', s']`
  (centre, area, aspect ratio + velocities). Explicit predict/update, tuned
  process/measurement noise, covariance propagation. See `tracker/kalman.py`.
- **Association** — IoU cost matrix, optimal assignment via
  `scipy.optimize.linear_sum_assignment`; IoU gating rejects weak matches;
  greedy fallback if scipy is unavailable.
- **Lifecycle** — new detections start `TENTATIVE`; `n_init` consecutive hits
  → `CONFIRMED`; unmatched tracks coast (KF-only) up to `max_age` frames before
  `DELETED`. Handles occlusions/crossings without dropping IDs.
- **Re-ID (DeepSORT-lite)** — appearance embedding is intentionally **not**
  enabled (would add torch + risk). The association layer is written so an
  appearance-distance term can be added later; documented as future work.

---

## Real MOT17 benchmark — one command (labels auto-downloaded)

```bash
pip install numpy scipy motmetrics
python run_real.py
```

`run_real.py` is the turnkey real-data runner. It downloads **only the small
text label files** (`det/det.txt` + `gt/gt.txt`, a few MB) for four MOT17 train
sequences (MOT17-02/-04/-09/-10, FRCNN public detections) from a labels mirror —
**never the multi-GB image frames**, which tracking-by-detection does not need —
then runs SORT per sequence and scores it against ground truth with the
reference `motmetrics` library, printing a per-sequence + overall table
(MOTA / MOTP / IDF1 / ID-switches / MT / ML / FP / FN). See
[`RESULTS.md`](RESULTS.md) §"REAL — MOT17" for the actual measured numbers.

Scaling: add more sequences with `--seqs MOT17-05-SDP ...`, or point at a full
local MOTChallenge download; sweep the detection gate with `--det-score`.

## Run a single sequence you already have (MOT16 / MOT17)

`run_mot16.py` scores one sequence folder you point it at. To get headline
numbers manually:

1. **Download** the free MOTChallenge data from <https://motchallenge.net>
   (MOT16 or MOT17). Extract so a sequence looks like:
   ```
   MOT16/train/MOT16-02/det/det.txt      # public detections
   MOT16/train/MOT16-02/gt/gt.txt        # ground truth (train split only)
   ```
2. **Run + score** a train sequence (uses the provided public detections):
   ```bash
   python run_mot16.py --seq path/to/MOT16/train/MOT16-02
   ```
   Prints real MOTA / MOTP / IDF1 / ID-switches for that sequence.
3. **Produce a submission** for a test sequence (no gt available locally):
   ```bash
   python run_mot16.py --seq path/to/MOT16/test/MOT16-01 --out results/MOT16-01.txt
   ```
   Submit the results file to MOTChallenge for official scoring.

Using a **learned detector** instead of the public detections:
`tracker/detectors.py` has `YoloDetector` (Ultralytics). Install
`pip install ultralytics`, run YOLO per frame, and feed its `[x1,y1,x2,y2,score]`
output to `Sort.update()`.

---

## Honesty note
All numbers in `RESULTS.md` come from **running this code**. The smoke-test
numbers are on a **controlled synthetic sequence** and are labelled as such —
they are **not** MOT16 results. Use `run_mot16.py` for real-benchmark numbers.
