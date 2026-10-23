"""Shared helpers: paths, crops, keypoint-model inference, and the length metrics.

Crop rule (train, val and test alike): a square centred on the SAM 3.1 mask box of the fish under the
laser dot (as production picks the fish), side = BOX_SCALE x box diagonal + CROP_PAD, clipped to the
frame (crop_from_box). Training frames get their box from SAM run on the stored crop
(sam_boxes.py --on-crops), which matches the full-window box closely on val (median IoU 0.985).
Training frames with no human dot or no mask keep the stored click-centred crop
(square_crop: CROP_SCALE x head-tail length + CROP_PAD).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CSCW = Path(os.environ.get("CSCW_REPO", HERE.parent / "cscw-fishsense2027")).resolve()
DATA = Path(os.environ.get("HEADTAIL_DATA", f"/local/{os.environ.get('USER', 'user')}/headtail_keypoints"))
STAGE = DATA / "frames"
YOLO = DATA / "yolo"
CROP_SCALE, CROP_PAD = 1.6, 128   # stage.py uses the same constants for the stored training crops
# Model crops are cut from the SAM 3.1 mask box at every split: side = BOX_SCALE x box diagonal + CROP_PAD.
# The box diagonal is ~1.1x the head-tail length, so 1.4 keeps the crop inside the stored training crop
# (1.6 x length + 128) almost always; what falls outside is padded grey.
BOX_SCALE = 1.4
PAD_VALUE = 114                    # Ultralytics' letterbox grey
BODY_HEIGHT = 0.35                 # box height / length for the training box label (the box is not scored)


def square_crop(cx: float, cy: float, extent: float, W: int, H: int, scale: float = CROP_SCALE) -> tuple[int, int, int, int]:
    half = 0.5 * (scale * extent + CROP_PAD)
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    return x0, y0, min(W, int(np.ceil(cx + half))), min(H, int(np.ceil(cy + half)))


def crop_from_box(bbox, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return square_crop(0.5 * (x0 + x1), 0.5 * (y0 + y1), float(np.hypot(x1 - x0, y1 - y0)), W, H, scale=BOX_SCALE)


def box_from_keypoints(hx, hy, tx, ty) -> tuple[float, float, float, float]:
    """Axis-aligned box of a head-tail rectangle (5% past each end, BODY_HEIGHT x length tall)."""
    d = np.array([tx - hx, ty - hy], float); L = float(np.hypot(*d)); u = d / max(L, 1e-9); n = np.array([-u[1], u[0]])
    a, b = np.array([hx, hy]) - 0.05 * L * u, np.array([tx, ty]) + 0.05 * L * u
    h = 0.5 * BODY_HEIGHT * L * n
    P = np.stack([a + h, a - h, b + h, b - h])
    return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())


def yolo_label(crop_w, crop_h, hx, hy, tx, ty) -> str:
    """One YOLO-pose line in crop coordinates: class 0, box, keypoint 0 = snout (head), 1 = fork (tail)."""
    x0, y0, x1, y1 = box_from_keypoints(hx, hy, tx, ty)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(crop_w, x1), min(crop_h, y1)
    f = lambda v, s: f"{min(max(v / s, 0.0), 1.0):.6f}"
    return " ".join(["0", f(0.5 * (x0 + x1), crop_w), f(0.5 * (y0 + y1), crop_h), f(x1 - x0, crop_w), f(y1 - y0, crop_h),
                     f(hx, crop_w), f(hy, crop_h), "2", f(tx, crop_w), f(ty, crop_h), "2"])


def pick(result, dot_xy=None):
    """Keypoints (head_x, head_y, tail_x, tail_y, conf) of the detection containing the dot (crop coords),
    else the most confident; None if nothing was detected."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    B = result.boxes.xyxy.cpu().numpy(); C = result.boxes.conf.cpu().numpy(); K = result.keypoints.xy.cpu().numpy()
    i = int(np.argmax(C))
    if dot_xy is not None:
        inside = [j for j in range(len(B)) if B[j, 0] <= dot_xy[0] <= B[j, 2] and B[j, 1] <= dot_xy[1] <= B[j, 3]]
        if inside:
            i = max(inside, key=lambda j: C[j])
    return float(K[i, 0, 0]), float(K[i, 0, 1]), float(K[i, 1, 0]), float(K[i, 1, 1]), float(C[i])


def keypoint_errors(df, pre: str, ref: str = "") -> dict:
    """Per-frame length ratio and point errors of (pre head/tail) against the human clicks, as fractions of
    the human length. Orientation-free point errors match each predicted point to the nearer human end."""
    hx, hy, tx, ty = (df[f"{ref}{c}"].to_numpy(float) for c in ("head_x", "head_y", "tail_x", "tail_y"))
    px, py, qx, qy = (df[f"{pre}{c}"].to_numpy(float) for c in ("head_x", "head_y", "tail_x", "tail_y"))
    L = np.hypot(hx - tx, hy - ty)
    d = np.hypot(px - qx, py - qy) / L - 1
    same = np.hypot(px - hx, py - hy) + np.hypot(qx - tx, qy - ty)
    swap = np.hypot(px - tx, py - ty) + np.hypot(qx - hx, qy - hy)
    swapped = swap < same
    snout = np.where(swapped, np.hypot(qx - hx, qy - hy), np.hypot(px - hx, py - hy)) / L
    fork = np.where(swapped, np.hypot(px - tx, py - ty), np.hypot(qx - tx, qy - ty)) / L
    return dict(d=d, snout=snout, fork=fork, swapped=swapped)
