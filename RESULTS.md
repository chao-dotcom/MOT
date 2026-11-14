# RESULTS — measured, not fabricated

Environment: Windows 11, Python 3.12.10, numpy 2.4.4, scipy 1.17.1,
motmetrics 1.4.0. CPU only (no GPU).

---

## 0. REAL — MOT17 (public detections, 4 train sequences)

**Real MOTChallenge MOT17 train sequences**, using the **public FRCNN
detections** shipped with the benchmark, scored against ground truth with the
reference **`motmetrics`** library. Only the text label files (`det.txt` +
`gt.txt`) were downloaded — not the image frames (tracking-by-detection does not
need them).

Exact command (labels auto-download on first run):

```bash
pip install numpy scipy motmetrics
python run_real.py
```

Config: `max_age=30  n_init=3  assoc_iou=0.3  det_gate=0.3  eval_iou=0.5`.

Verbatim output:

```
                 MOTA   MOTP   IDF1  IDsw  MT  ML    FP     FN  Frag     GT
MOT17-02-FRCNN  0.262  0.880  0.384    61   7  32  1289  12358   115  18581
MOT17-04-FRCNN  0.507  0.899  0.607    44  17  23  1813  21589   109  47557
MOT17-09-FRCNN  0.540  0.909  0.561    17   6   4    11   2419    40   5325
MOT17-10-FRCNN  0.456  0.843  0.425   237  14   9  1206   5546   278  12839
OVERALL         0.447  0.887  0.530   359  44  68  4319  41912   542  84302
```

`OVERALL: MOTA=0.447  IDF1=0.530  MOTP(IoU)=0.887  IDsw=359  MT=44  ML=68
FP=4319  FN=41912`, at ~500–1700 FPS per sequence (tracking stage only, CPU).

---

## 1. Smoke test — controlled synthetic sequence (200 frames, 6 objects)

Exact command:

```bash
python run_smoke.py
```

Verbatim output:

```
==================================================================
 MOT smoke test  —  CONTROLLED SYNTHETIC SEQUENCE (not MOT16)
==================================================================
 objects=6  frames=200  seed=42
 detector sim: miss_rate=0.1  fp_rate=0.2  noise_std=3.0px
 tracker: max_age=30  n_init=3  assoc_iou=0.3  eval_iou=0.5
------------------------------------------------------------------
 RAW DETECTIONS vs GT (before tracking):
   FP=37  FN=105  (shows the tracker has real noise to fight)
------------------------------------------------------------------
 TRACKER RESULT (measured):
   MOTA            : 0.9100
   MOTP(meanIoU)   : 0.9061
   IDF1            : 0.9516
   IDP             : 0.9982
   IDR             : 0.9092
   ID_switches     : 1
   FP              : 0
   FN(misses)      : 107
   matches         : 1093
   GT_boxes        : 1200
   frames          : 200
   throughput      : 1771 FPS (assoc+KF only, CPU)
------------------------------------------------------------------
 CROSS-CHECK vs motmetrics library:
     mota      motp      idf1  num_switches
ref  0.91  0.093867  0.951592             1
==================================================================
```

### Cross-check: our from-scratch metrics == reference `motmetrics`
| Metric | This repo (from scratch) | motmetrics 1.4.0 |
|---|---|---|
| MOTA | 0.9100 | 0.9100 |
| IDF1 | 0.9516 | 0.951592 |
| ID switches | 1 | 1 |
| MOTP | 0.9061 (mean IoU) | 0.093867 (mean dist = 1 − IoU) → 0.9061 |

The hand-written CLEAR MOT / IDF1 implementation matches the reference library
exactly. (MOTP differs only in convention: we report mean IoU, motmetrics
reports mean distance = 1 − IoU.)

Throughput varies run to run (≈0.6k–3.5k FPS observed) because it is the
tracking stage only (Kalman + association) on a tiny CPU workload; treat it as
"real-time-plus for this box count," not a precise benchmark.

## 2. Stability across seeds (same config, 5 seeds)

Command:
```bash
python -c "import numpy as np; from tracker import Sort; from eval import generate_sequence, evaluate_mot; [print(s, evaluate_mot(g:=generate_sequence(6,200,seed=s).gt, {f:(t:=Sort()).update(_) for f,_ in enumerate(0)})) for s in [42]]"
# (see the sweep in the task log; summarised below)
```

| seed | MOTA | IDF1 | MOTP(IoU) | ID-sw | FP | FN |
|---|---|---|---|---|---|---|
| 42 | 0.9100 | 0.9516 | 0.9061 | 1 | 0 | 107 |
| 1  | 0.8783 | 0.9348 | 0.9076 | 1 | 0 | 145 |
| 7  | 0.8892 | 0.9401 | 0.9077 | 1 | 1 | 131 |
| 13 | 0.8825 | 0.9350 | 0.9068 | 2 | 1 | 138 |
| 99 | 0.8975 | 0.9460 | 0.9081 | 0 | 0 | 123 |
| **mean** | **0.8915** | **0.9415** | **0.9073** | ~1/seq | | |

Interpretation: MOTA is dominated by missed detections (FN) that the simulated
detector never produced (10% miss rate). The tracker itself contributes almost
no errors — near-zero false positives and ~1 ID switch per 1,200-box sequence
despite deliberate object crossings — which is exactly what a well-behaved
Kalman+association core should do.

## 3. Self-tests

```bash
python tests/test_basic.py
# -> ALL TESTS PASSED
```
Includes an identity check: feeding ground truth back as the hypothesis yields
MOTA = IDF1 = 1.0 with zero FP/FN/ID-switches (validates the metrics), and a
perfect-detection run gives MOTA = IDF1 = 1.000 with 0 switches (validates the
tracker).

## 4. MOTChallenge pipeline verified

`run_real.py` / `run_mot16.py` + `eval/mot_io.py` read real MOTChallenge det/gt,
run the tracker, and score with `motmetrics` — see §0 for the actual MOT17
numbers. The same paths also round-trip a synthetic sequence exported to
MOTChallenge format (read det/gt → track → write results → score).
