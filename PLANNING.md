# MOT — Planning Doc (for the implementing agent)

## What this repo is for

This is Chao Hsuan Ho's **public-facing GitHub repo** for the résumé PROJECT
bullet:

> **Multi-Object Tracker — tracking-by-detection (from scratch)** *(Solo ·
> Python / NumPy · 2026)*
> - Built a SORT-style MOT pipeline — per-track constant-velocity **Kalman
>   filter** + **Hungarian IoU data association** + track lifecycle — in
>   NumPy/SciPy (no tracking libraries).
> - Benchmarked on **real MOT17 train sequences (public detections, ~84k
>   boxes)**: **MOTA 0.45 / IDF1 0.53 / MOTP 0.89**, scored with the
>   **`motmetrics`** reference library — in line with the published SORT
>   baseline; error is detector-recall-bound (FN ≫ FP), 500–1700 FPS on CPU.

Right now this repo (`github.com/chao-dotcom/MOT`) is **empty except an
initial commit** — no code, no docs. That's the gap this planning doc closes.

**This is why the project exists at all**: it is the résumé's *only*
non-deployment, pure-algorithm proof point (everything else — the Logitech
SR/detection pipeline, QuantaVision — is "picked/deployed a model," not
"derived the algorithm"). The project's own tracking log flags this
explicitly:

> "You must study, run, and be able to defend `mot_tracker/` line-by-line
> before putting them on a résumé... if you can't explain the Kalman/Hungarian
> logic or the ISP stage order + noise model in an interview, that's the same
> fabrication trap. Run them, read them, tweak something, make them yours."

**Implication for how you build this**: don't just make it work — make it
*legible*. Comments and README prose should explain *why* a design choice was
made (state-vector parameterization, gating threshold, lifecycle constants),
not just restate the code, because Chao has to be able to reconstruct and
defend every one of these choices out loud in an interview. Optimize for
"a human can read this once and explain it back," not for cleverness.

## Reference implementation already exists — use it as ground truth

A working, already-benchmarked version of this exact project lives locally
(not in this repo) at:

```
C:\Users\cho2\Desktop\folders\job-search\projects\mot_tracker\
```

It already runs, already passes its own self-tests, and already has real
measured numbers (below). **Treat it as the spec / oracle for algorithm
behavior and exact numbers** — this repo's implementation should reproduce
the same design and the same measured results, not invent a different
tracker. Whether you port the code over directly or rewrite it cleanly for
public presentation, the acceptance bar is: **same algorithm, same numbers,
verifiably reproducible from this repo alone** (nothing that only exists on
Chao's machine).

Do not copy files in blindly without reading them — the whole point of this
repo is that the code has to end up something a person (not just an agent)
understands.

## Algorithm design spec

**Tracking-by-detection, SORT-style, online (frame-by-frame, no batch/offline
re-ranking):**

- **State estimation** — one **constant-velocity Kalman filter per track**,
  7-D state `[u, v, s, r, u', v', s']` (bbox center x/y, area, aspect ratio,
  plus velocities of the first three). Explicit predict/update step each
  frame; tuned process/measurement noise; full covariance propagation.
- **Data association** — per-frame **IoU cost matrix** between predicted
  track boxes and new detections; optimal assignment via the **Hungarian
  algorithm** (`scipy.optimize.linear_sum_assignment`), with a dependency-free
  **greedy fallback** if `scipy` isn't available. An **IoU gate** (reject
  matches below a threshold, default `0.3`) prevents nonsense matches.
- **Track lifecycle** — three states: `TENTATIVE` (just born) →
  `CONFIRMED` (after `n_init` consecutive hits, default `3`) → coasts
  KF-only when unmatched → `DELETED` after `max_age` (default `30`) missed
  frames. This is what lets the tracker survive brief occlusions/crossings
  without dropping an ID.
- **Detector input is decoupled from the tracker** — the tracker consumes
  `[x1, y1, x2, y2, score]` boxes from *any* source: MOTChallenge's public
  `det.txt`, or a real detector (an `ultralytics` YOLO wrapper is in-scope as
  an optional plug-in, not required for the core benchmark).
- **Explicitly out of scope for this repo** (document as "future work," don't
  implement): appearance re-ID / DeepSORT-style embedding matching. The
  association layer should be structured so an appearance-distance term could
  be added later, but adding it now is scope creep this project doesn't need.

## Evaluation plan — from-scratch metrics, cross-validated against a reference library

Two evaluation modes, kept clearly separated (never blur them):

1. **Controlled synthetic sequence** (known ground truth you generate
   yourself: objects at constant velocity, bouncing off frame edges, crossing
   paths; a simulated noisy detector with a miss rate, false-positive rate,
   and localization noise). Used for fast iteration and self-tests — label
   every number from this as "synthetic," never as a benchmark result.
2. **Real MOT17 train sequences, public detections** — the actual benchmark
   claim. Download **only the text label files** (`det.txt` + `gt.txt`, a few
   MB per sequence) from MOTChallenge — never the multi-GB image frames,
   which tracking-by-detection doesn't need. Run the tracker on the public
   FRCNN detections for at least MOT17-02/-04/-09/-10, and score against
   ground truth.

**Metrics must be implemented from scratch** (CLEAR MOT: MOTA, MOTP, FP, FN,
ID-switches; plus IDF1) — this is itself part of the algorithm-depth claim.
Then **cross-validate every number against the `motmetrics` reference
library** and report both side by side; they must match (mean IoU vs. mean
distance is the only expected convention difference for MOTP).

### Target numbers (must reproduce these, or something in the same neighborhood with an honest explanation of any delta)

**Real MOT17** (config: `max_age=30 n_init=3 assoc_iou=0.3 det_gate=0.3
eval_iou=0.5`):

```
                 MOTA   MOTP   IDF1  IDsw  MT  ML    FP     FN  Frag     GT
MOT17-02-FRCNN  0.262  0.880  0.384    61   7  32  1289  12358   115  18581
MOT17-04-FRCNN  0.507  0.899  0.607    44  17  23  1813  21589   109  47557
MOT17-09-FRCNN  0.540  0.909  0.561    17   6   4    11   2419    40   5325
MOT17-10-FRCNN  0.456  0.843  0.425   237  14   9  1206   5546   278  12839
OVERALL         0.447  0.887  0.530   359  44  68  4319  41912   542  84302
```

~500–1700 FPS per sequence (tracking stage only, CPU, no GPU).

**Synthetic smoke test** (200 frames, 6 objects, seed 42, miss_rate=0.1,
fp_rate=0.2, noise_std=3.0px): MOTA 0.9100, MOTP(meanIoU) 0.9061, IDF1
0.9516, ID-switches 1, matching `motmetrics` to 4 decimal places.

**Why MOT17 MOTA (0.447) is much lower than the synthetic MOTA (0.91), and
that's correct, not a bug**: SORT on MOT17 *public* detections lands in the
~40–60 MOTA range in the published literature (e.g. ByteTrack paper reports
SORT ≈ MOTA 43.1 on the public-detection protocol) — error is
**detector-recall-bound** (FN=41,912 ≫ FP=4,319): the tracker only tracks
what the public detector found, and the FRCNN public set misses a lot of
pedestrians. MOTP(IoU)=0.887 shows matched boxes are tight — the
Kalman/association core itself localizes well. **Never claim a leaderboard
number** — these are raw public detections + pedestrian-class GT without the
official MOTChallenge distractor/ignore-region preprocessing, so state this
as "a reproducible, honestly-scored local result," not an official score.

### Self-test requirement

Include an identity check: feeding ground truth back in as the hypothesis
must yield MOTA = IDF1 = 1.0 with zero FP/FN/ID-switches (this is what
validates the metrics code itself, independent of the tracker).

## Repo structure to build

```
tracker/
  kalman.py        Constant-velocity Kalman filter (7-D state), one per track
  association.py   Vectorized IoU + Hungarian (scipy) / greedy fallback + gating
  track.py         Track lifecycle: TENTATIVE -> CONFIRMED -> coast -> DELETED
  sort.py          Sort: online tracker tying the above together
  detectors.py     FileDetector (MOT det.txt) + optional YoloDetector (ultralytics)
eval/
  synthetic.py     Controlled sequence generator with known GT + simulated detector
  metrics.py       From-scratch CLEAR MOT (MOTA/MOTP, FP/FN/IDSW) + IDF1
  mot_io.py        Read/write MOTChallenge (MOT16/17) det/gt/results files
run_smoke.py       One command -> measured metrics on the synthetic sequence
run_real.py        Turnkey: auto-downloads MOT17 label files, runs + scores real data
run_mot16.py       Point-at-a-local-sequence entry point (manual MOTChallenge download)
tests/test_basic.py  Dependency-free self-tests (IoU, KF, metrics identity)
README.md          Quick start + algorithm notes + honesty note
RESULTS.md         Verbatim measured output for every claim above
requirements.txt
LICENSE
```

## Deliverables checklist for the implementing agent

- [ ] `tracker/` — Kalman filter, IoU+Hungarian association with gating,
      track lifecycle, the `Sort` class, detector adapters
- [ ] `eval/` — synthetic sequence generator, from-scratch CLEAR-MOT/IDF1
      metrics, MOTChallenge I/O
- [ ] `run_smoke.py` — zero-download synthetic demo, runs in ~1 second
- [ ] `run_real.py` — one-command real MOT17 benchmark (auto-downloads only
      the small label files, never image frames), scores with `motmetrics`
- [ ] `run_mot16.py` — manual-download single-sequence entry point +
      submission-file output mode
- [ ] `tests/test_basic.py` — self-tests including the GT-identity check
- [ ] `README.md` — quick start, "what's inside," algorithm notes explaining
      *why* each design choice was made, and an explicit **honesty note**
      (synthetic numbers are never presented as benchmark results)
- [ ] `RESULTS.md` — verbatim command + verbatim output for every number
      claimed, plus the motmetrics cross-check table, plus the
      "why MOT17 MOTA is lower than synthetic MOTA" explanation
- [ ] `requirements.txt` (numpy, scipy; motmetrics + ultralytics optional)
- [ ] `LICENSE`
- [ ] Verify every number in `RESULTS.md` by actually running the commands
      before committing them — no number goes in that wasn't just produced
      by running this repo's own code

## Grounding rules (same discipline as the rest of Chao's job-search repo — do not violate)

- Every number in `RESULTS.md` must come from actually running the code in
  *this* repo — no numbers carried over from memory or from the reference
  implementation without re-verifying here.
- Label synthetic vs. real results unambiguously everywhere they appear
  (README, RESULTS.md, code comments, commit messages) — never let a
  synthetic number be mistaken for a MOT17 result.
- Never claim an official MOTChallenge leaderboard submission/score — these
  are local, honestly-scored numbers using public detections without the
  official distractor/ignore-region preprocessing.
- No appearance re-ID / DeepSORT claim — it isn't built. Document it as
  future work only, in a way the association code structurally allows later.
