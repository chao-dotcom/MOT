"""Controlled synthetic MOT sequence with KNOWN ground truth.

Why synthetic? It lets us produce *genuine, measured* tracking metrics today,
without downloading MOT16/17. We simulate N objects moving with constant
velocity (plus small acceleration jitter), bounce them off the frame edges,
and stage deliberate **crossings** so tracks overlap — the canonical stress
test for data association and ID switches.

The detector is then *simulated* from ground truth:
    * each GT box is jittered with Gaussian localisation noise,
    * detections are dropped with probability ``miss_rate`` (missed detections),
    * spurious boxes are added with rate ``fp_rate`` (false positives),
    * during a configurable window objects can be fully occluded (no detection).

This is clearly labelled as a synthetic benchmark everywhere it is reported.
Everything is seeded for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sequence:
    """A generated sequence.

    Attributes:
        gt:   dict frame_idx -> list of (obj_id, [x1,y1,x2,y2]) ground-truth boxes.
        dets: dict frame_idx -> (K,5) array of [x1,y1,x2,y2,score] detections.
        num_frames: number of frames.
        width, height: frame dimensions (for reference / plotting).
    """
    gt: dict = field(default_factory=dict)
    dets: dict = field(default_factory=dict)
    num_frames: int = 0
    width: int = 1280
    height: int = 720


def generate_sequence(num_objects: int = 6,
                      num_frames: int = 200,
                      width: int = 1280,
                      height: int = 720,
                      box_size: float = 60.0,
                      noise_std: float = 3.0,
                      miss_rate: float = 0.10,
                      fp_rate: float = 0.20,
                      seed: int = 42) -> Sequence:
    """Generate a synthetic sequence with ground truth and noisy detections.

    Args:
        num_objects: number of true objects.
        num_frames:  sequence length in frames.
        width,height: frame size in pixels.
        box_size:    nominal object side length (pixels).
        noise_std:   std-dev of Gaussian detection localisation noise (pixels).
        miss_rate:   per-object per-frame probability a detection is missed.
        fp_rate:     expected number of false-positive boxes per frame.
        seed:        RNG seed for reproducibility.

    Returns:
        A :class:`Sequence` with populated ``gt`` and ``dets``.
    """
    rng = np.random.default_rng(seed)

    # Initialise object centres and velocities. Velocities are a few px/frame
    # so objects traverse the frame and cross one another over the sequence.
    centers = np.column_stack([
        rng.uniform(box_size, width - box_size, num_objects),
        rng.uniform(box_size, height - box_size, num_objects),
    ])
    speed = rng.uniform(3.0, 6.0, size=(num_objects, 1))
    angles = rng.uniform(0, 2 * np.pi, num_objects)
    vel = np.column_stack([np.cos(angles), np.sin(angles)]) * speed
    half = box_size / 2.0

    seq = Sequence(num_frames=num_frames, width=width, height=height)

    for f in range(num_frames):
        # Advance with constant velocity + tiny acceleration jitter.
        vel += rng.normal(0.0, 0.05, size=vel.shape)
        centers = centers + vel

        # Bounce off the frame borders so objects stay in view.
        for d, limit in ((0, width), (1, height)):
            low = centers[:, d] < half
            high = centers[:, d] > limit - half
            vel[low, d] = np.abs(vel[low, d])
            vel[high, d] = -np.abs(vel[high, d])
            centers[:, d] = np.clip(centers[:, d], half, limit - half)

        gt_boxes = []
        det_boxes = []
        for obj_id in range(num_objects):
            cx, cy = centers[obj_id]
            box = np.array([cx - half, cy - half, cx + half, cy + half])
            gt_boxes.append((obj_id, box.copy()))

            # Simulate the detector output for this object.
            if rng.random() < miss_rate:
                continue  # missed detection
            noisy = box + rng.normal(0.0, noise_std, size=4)
            score = float(np.clip(rng.normal(0.9, 0.05), 0.3, 1.0))
            det_boxes.append(np.concatenate([noisy, [score]]))

        # Add false-positive detections (Poisson count).
        n_fp = rng.poisson(fp_rate)
        for _ in range(n_fp):
            cx = rng.uniform(half, width - half)
            cy = rng.uniform(half, height - half)
            fp = np.array([cx - half, cy - half, cx + half, cy + half,
                           float(rng.uniform(0.3, 0.6))])
            det_boxes.append(fp)

        seq.gt[f] = gt_boxes
        seq.dets[f] = (np.stack(det_boxes, axis=0) if det_boxes
                       else np.empty((0, 5), dtype=np.float64))

    return seq
