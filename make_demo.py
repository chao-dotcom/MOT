"""Generate demo.gif — a visual side-by-side of raw noisy detections vs the
SORT tracker's output on the same controlled synthetic sequence used in
RESULTS.md (6 objects, 200 frames, seed=42).

Left panel:  raw simulated detections (no identity, flicker with misses/FPs).
Right panel: SORT tracker output (persistent colored ID + short motion trail),
             showing IDs surviving the deliberate crossings/occlusions.

Run:  python make_demo.py   ->  writes demo.gif to the repo root.
"""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle

from eval import generate_sequence
from tracker import Sort

NUM_OBJECTS = 6
NUM_FRAMES = 200
SEED = 42
STRIDE = 2          # subsample frames -> smaller/faster gif
TRAIL_LEN = 15       # frames of motion trail to draw per track
FPS = 12

seq = generate_sequence(NUM_OBJECTS, NUM_FRAMES, seed=SEED)
tracker = Sort(max_age=30, n_init=3, iou_threshold=0.3)

tracked_per_frame = {}
for f in range(seq.num_frames):
    tracked_per_frame[f] = tracker.update(seq.dets[f]).copy()

colors = plt.get_cmap("tab10").colors
trails = defaultdict(list)

fig, (ax_raw, ax_trk) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=100)
for ax, title in ((ax_raw, "Raw noisy detections (no tracking)"),
                   (ax_trk, "SORT tracker output (persistent ID)")):
    ax.set_xlim(0, seq.width)
    ax.set_ylim(seq.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
fig.suptitle("MOT tracker demo — synthetic sequence, 6 objects, 200 frames (seed=42)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

writer = PillowWriter(fps=FPS)
with writer.saving(fig, "demo.gif", dpi=100):
    for f in range(0, seq.num_frames, STRIDE):
        for ax in (ax_raw, ax_trk):
            for artist in list(ax.patches) + list(ax.texts) + list(ax.lines):
                artist.remove()

        # Left: raw detections, gray, no identity.
        for det in seq.dets[f]:
            x1, y1, x2, y2 = det[:4]
            ax_raw.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor="0.5", linewidth=1.5))
        ax_raw.text(10, 25, f"frame {f:03d}  dets={len(seq.dets[f])}",
                    fontsize=9, color="0.3")

        # Right: tracker output, colored + persistent ID + trail.
        for x1, y1, x2, y2, tid in tracked_per_frame[f]:
            tid = int(tid)
            color = colors[tid % len(colors)]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            trails[tid].append((cx, cy))
            trails[tid] = trails[tid][-TRAIL_LEN:]

            ax_trk.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=color, linewidth=2))
            ax_trk.text(x1, y1 - 4, f"ID {tid}", fontsize=8, color=color,
                        fontweight="bold")
            if len(trails[tid]) > 1:
                xs, ys = zip(*trails[tid])
                ax_trk.plot(xs, ys, color=color, linewidth=1, alpha=0.6)
        ax_trk.text(10, 25, f"frame {f:03d}  tracks={len(tracked_per_frame[f])}",
                    fontsize=9, color="0.3")

        writer.grab_frame()

plt.close(fig)
print("wrote demo.gif")
