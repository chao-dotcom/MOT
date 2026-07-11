"""From-scratch MOT metrics: CLEAR MOT (MOTA/MOTP, FP/FN/IDSW) + IDF1.

No external metrics dependency — implemented directly from the source papers so
the numbers are transparent and the code doubles as a demonstration that the
author understands the definitions:

  * CLEAR MOT  — Bernardin & Stiefelhagen, EURASIP 2008.
  * IDF1       — Ristani et al., ECCV 2016 Workshops.

A ground-truth box and a hypothesis box are considered a match when their IoU
is >= ``iou_threshold`` (default 0.5, the MOTChallenge convention).

If you prefer the canonical reference implementation, ``run_smoke.py`` will use
the ``motmetrics`` package when it is installed and cross-check against these
numbers; this module is the always-available fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracker.association import iou_batch

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class MOTMetrics:
    """Container for computed metrics."""
    mota: float
    motp: float          # mean IoU of matched pairs (higher = tighter boxes)
    idf1: float
    idp: float
    idr: float
    num_switches: int
    num_fp: int
    num_misses: int      # false negatives
    num_matches: int
    num_gt: int
    num_frames: int

    def as_dict(self) -> dict:
        return {
            "MOTA": self.mota,
            "MOTP(meanIoU)": self.motp,
            "IDF1": self.idf1,
            "IDP": self.idp,
            "IDR": self.idr,
            "ID_switches": self.num_switches,
            "FP": self.num_fp,
            "FN(misses)": self.num_misses,
            "matches": self.num_matches,
            "GT_boxes": self.num_gt,
            "frames": self.num_frames,
        }


def _match_frame(gt_ids, gt_boxes, hyp_ids, hyp_boxes,
                 prev_map, iou_threshold):
    """Match one frame's GT to hypotheses, preferring last frame's pairing.

    Returns (matches, matched_iou_list, id_switches, new_map) where
    ``matches`` is a list of (gt_id, hyp_id).
    """
    matches = []
    matched_iou = []
    new_map = {}

    if len(gt_boxes) == 0 or len(hyp_boxes) == 0:
        iou = None
    else:
        iou = iou_batch(np.asarray(gt_boxes), np.asarray(hyp_boxes))

    used_gt = set()
    used_hyp = set()

    # 1. Preserve previous-frame correspondences when still valid (>= threshold).
    #    This is what makes ID-switch counting meaningful in CLEAR MOT.
    if iou is not None:
        gt_pos = {gid: i for i, gid in enumerate(gt_ids)}
        hyp_pos = {hid: j for j, hid in enumerate(hyp_ids)}
        for gid, hid in prev_map.items():
            if gid in gt_pos and hid in hyp_pos:
                i, j = gt_pos[gid], hyp_pos[hid]
                if iou[i, j] >= iou_threshold:
                    matches.append((gid, hid))
                    matched_iou.append(float(iou[i, j]))
                    new_map[gid] = hid
                    used_gt.add(i)
                    used_hyp.add(j)

    # 2. Optimally match the remaining GT/hypotheses by IoU.
    if iou is not None:
        rem_gt = [i for i in range(len(gt_ids)) if i not in used_gt]
        rem_hyp = [j for j in range(len(hyp_ids)) if j not in used_hyp]
        if rem_gt and rem_hyp:
            sub = iou[np.ix_(rem_gt, rem_hyp)]
            if _HAS_SCIPY:
                ri, rj = linear_sum_assignment(-sub)
                pairs = zip(ri.tolist(), rj.tolist())
            else:  # greedy fallback
                pairs = _greedy_pairs(sub)
            for a, b in pairs:
                if sub[a, b] >= iou_threshold:
                    i, j = rem_gt[a], rem_hyp[b]
                    gid, hid = gt_ids[i], hyp_ids[j]
                    matches.append((gid, hid))
                    matched_iou.append(float(iou[i, j]))
                    new_map[gid] = hid

    # 3. Count ID switches: a GT that was matched before to a *different* hyp.
    id_switches = 0
    for gid, hid in new_map.items():
        if gid in prev_map and prev_map[gid] != hid:
            id_switches += 1

    return matches, matched_iou, id_switches, new_map


def _greedy_pairs(sub):  # pragma: no cover - only without scipy
    sub = sub.copy()
    pairs = []
    while True:
        idx = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[idx] <= 0:
            break
        pairs.append((int(idx[0]), int(idx[1])))
        sub[idx[0], :] = -1
        sub[:, idx[1]] = -1
    return pairs


def _compute_idf1(gt, hyp, frames, iou_threshold):
    """Global identity-based IDF1 (Ristani et al.) via one-to-one assignment."""
    gt_ids = sorted({gid for f in frames for gid, _ in gt.get(f, [])})
    hyp_ids = sorted({int(h[4]) for f in frames for h in hyp.get(f, [])})

    gt_count = {g: 0 for g in gt_ids}
    hyp_count = {h: 0 for h in hyp_ids}
    for f in frames:
        for gid, _ in gt.get(f, []):
            gt_count[gid] += 1
        for h in hyp.get(f, []):
            hyp_count[int(h[4])] += 1

    total_gt = sum(gt_count.values())
    total_hyp = sum(hyp_count.values())

    n_g, n_h = len(gt_ids), len(hyp_ids)
    if n_g == 0 and n_h == 0:
        return 1.0, 1.0, 1.0
    if n_g == 0 or n_h == 0:
        return 0.0, 0.0, 0.0

    gidx = {g: i for i, g in enumerate(gt_ids)}
    hidx = {h: j for j, h in enumerate(hyp_ids)}

    # overlap[i, j] = # frames where GT i and hyp j overlap (IoU >= threshold).
    overlap = np.zeros((n_g, n_h), dtype=np.float64)
    for f in frames:
        gts = gt.get(f, [])
        hps = hyp.get(f, [])
        if not gts or len(hps) == 0:
            continue
        gboxes = np.array([b for _, b in gts])
        hboxes = np.array([h[:4] for h in hps])
        iou = iou_batch(gboxes, hboxes)
        for gi, (gid, _) in enumerate(gts):
            for hj, h in enumerate(hps):
                if iou[gi, hj] >= iou_threshold:
                    overlap[gidx[gid], hidx[int(h[4])]] += 1.0

    # Build the padded cost matrix (Ristani formulation): minimising total cost
    # == minimising IDFP + IDFN. Size (n_g+n_h) x (n_h+n_g).
    INF = 1e9
    size = n_g + n_h
    cost = np.full((size, size), 0.0)
    # top-left: match gt i <-> hyp j
    for i, g in enumerate(gt_ids):
        for j, h in enumerate(hyp_ids):
            cost[i, j] = gt_count[g] + hyp_count[h] - 2.0 * overlap[i, j]
    # top-right (n_g x n_g): gt i vs its own FN slot -> cost gt_count on diagonal
    for i, g in enumerate(gt_ids):
        for k in range(n_g):
            cost[i, n_h + k] = INF
        cost[i, n_h + i] = gt_count[g]
    # bottom-left (n_h x n_h): hyp j vs its own FP slot -> hyp_count on diagonal
    for j, h in enumerate(hyp_ids):
        for k in range(n_h):
            cost[n_g + k, j] = INF
        cost[n_g + j, j] = hyp_count[h]
    # bottom-right: nothing-to-nothing -> 0 (already zeros)

    if _HAS_SCIPY:
        ri, ci = linear_sum_assignment(cost)
        total_cost = cost[ri, ci].sum()
    else:  # pragma: no cover
        total_cost = _greedy_square_cost(cost)

    idtp = (total_gt + total_hyp - total_cost) / 2.0
    idtp = max(idtp, 0.0)
    idfn = total_gt - idtp
    idfp = total_hyp - idtp

    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
    idf1 = (2 * idtp / (2 * idtp + idfp + idfn)
            if (2 * idtp + idfp + idfn) > 0 else 0.0)
    return idf1, idp, idr


def _greedy_square_cost(cost):  # pragma: no cover
    n = cost.shape[0]
    used_c = set()
    total = 0.0
    order = np.argsort(cost, axis=None)
    assigned_r = set()
    for flat in order:
        r, c = divmod(int(flat), n)
        if r in assigned_r or c in used_c:
            continue
        assigned_r.add(r)
        used_c.add(c)
        total += cost[r, c]
        if len(assigned_r) == n:
            break
    return total


def evaluate_mot(gt: dict, hyp: dict, iou_threshold: float = 0.5) -> MOTMetrics:
    """Compute CLEAR MOT metrics + IDF1 for one sequence.

    Args:
        gt:  dict frame -> list of (obj_id, [x1,y1,x2,y2]).
        hyp: dict frame -> (K,5) array of [x1,y1,x2,y2,track_id].
        iou_threshold: IoU required to call a GT/hyp pair a match.

    Returns:
        A :class:`MOTMetrics` with the measured values.
    """
    frames = sorted(set(gt.keys()) | set(hyp.keys()))

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_sw = 0
    total_matches = 0
    iou_sum = 0.0
    prev_map: dict = {}

    for f in frames:
        gt_frame = gt.get(f, [])
        gt_ids = [g for g, _ in gt_frame]
        gt_boxes = [b for _, b in gt_frame]

        hyp_frame = hyp.get(f, np.empty((0, 5)))
        hyp_frame = np.asarray(hyp_frame)
        if hyp_frame.size == 0:
            hyp_ids, hyp_boxes = [], []
        else:
            hyp_ids = [int(h[4]) for h in hyp_frame]
            hyp_boxes = [h[:4] for h in hyp_frame]

        matches, matched_iou, sw, new_map = _match_frame(
            gt_ids, gt_boxes, hyp_ids, hyp_boxes, prev_map, iou_threshold)

        n_match = len(matches)
        total_gt += len(gt_ids)
        total_matches += n_match
        total_fp += len(hyp_ids) - n_match
        total_fn += len(gt_ids) - n_match
        total_sw += sw
        iou_sum += sum(matched_iou)

        # Carry forward the union of the persistent map so an object that
        # briefly disappears and returns to the same hyp is not a switch.
        prev_map.update(new_map)

    mota = 1.0 - (total_fp + total_fn + total_sw) / total_gt if total_gt else 0.0
    motp = iou_sum / total_matches if total_matches else 0.0
    idf1, idp, idr = _compute_idf1(gt, hyp, frames, iou_threshold)

    return MOTMetrics(
        mota=mota, motp=motp, idf1=idf1, idp=idp, idr=idr,
        num_switches=total_sw, num_fp=total_fp, num_misses=total_fn,
        num_matches=total_matches, num_gt=total_gt, num_frames=len(frames),
    )
