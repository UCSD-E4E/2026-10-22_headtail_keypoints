"""Build the keypoint datasets from the staged frames (CPU).

  train    YOLO-pose images/labels. Frames with a SAM 3.1 mask box (results/sam_boxes_crops.jsonl:
           SAM at the human dot on the stored crop) are cut by the deployment rule (hk.crop_from_box)
           from the stored crop, grey where the rule reaches past it; the rest keep the stored
           click-centred crop (<id>.crop.jpg). One fish per crop: box from the clicks
           (hk.box_from_keypoints), keypoint 0 = snout (human "head"), 1 = fork (human "tail").
           Writes results/train_crops.csv (source and padded fraction per frame).
  val      the same, from the staged full validation frames, cropped around the clicks the same way.
           Ultralytics' own validation (loss, pose mAP) runs on it; it is only a training monitor.
  sam      crops for the frames that have a SAM 3.1 mask box (results/sam_boxes.jsonl): every val
           frame at the human dot (the checkpoint-selection set, selected on LENGTH) and every test
           frame at the human and the automatic dot. Cropped by the mask box (hk.crop_from_box), as
           at deployment. Writes $HEADTAIL_DATA/yolo/sam/<id>_<seed>.jpg and results/sam_crops.csv.

Rebuilds from scratch each run (cheap: symlinks for train, crops for val/sam).
Run: uv run python scripts/build_dataset.py train val sam
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import hk  # noqa: E402

LOG = hk.HERE / "results/stage_log.jsonl"


def staged() -> pd.DataFrame:
    L = pd.DataFrame([json.loads(l) for l in LOG.open()])
    L = L[L.get("error").isna()] if "error" in L else L
    return L.drop_duplicates("image_id", keep="last")   # two staging runs overlapped on some frames; same bytes either way


def _train_crop(args):
    """One training image from a SAM box (worker): the deployment crop cut out of the stored crop."""
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)   # PR_SET_THP_DISABLE (see stage.py)
    cv2.setNumThreads(1)
    iid, box, stored, W, H = args
    x0, y0, x1, y1 = hk.crop_from_box(box, W, H); sx0, sy0, sx1, sy1 = stored
    src = cv2.imread(str(hk.STAGE / f"{iid}.crop.jpg"))
    out = np.full((y1 - y0, x1 - x0, 3), hk.PAD_VALUE, np.uint8)
    ax0, ay0, ax1, ay1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1)
    if ax1 > ax0 and ay1 > ay0:
        out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = src[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0]
    cv2.imwrite(str(hk.YOLO / "images" / os.environ.get("TRAIN_DIR", "train") / f"{iid}.jpg"), out)
    covered = max(0, ax1 - ax0) * max(0, ay1 - ay0) / ((x1 - x0) * (y1 - y0))
    return iid, (x0, y0, x1, y1), 1.0 - covered


def train_val(which: str):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    F = pd.read_csv(hk.HERE / "splits/frames.csv")
    S = staged().set_index("image_id")
    F = F[(F.split == which) & F.image_id.isin(S.index)]
    sub = os.environ.get("TRAIN_DIR", which) if which == "train" else which   # e.g. TRAIN_DIR=train_sam: build beside a set in use
    img, lab = hk.YOLO / "images" / sub, hk.YOLO / "labels" / sub
    for d in (img, lab):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    windows, rec = {}, []
    if which == "train":
        f = hk.HERE / "results/sam_boxes_crops.jsonl" if os.environ.get("TRAIN_CROPS", "sam") == "sam" else Path("/nonexistent")
        B = pd.DataFrame([json.loads(l) for l in f.open()]).drop_duplicates("image_id", keep="last") if f.exists() else pd.DataFrame(columns=["image_id", "split", "mask_bbox"])
        B = B[(B.split == "train") & B.mask_bbox.notna() & B.image_id.isin(F.image_id)]
        jobs = [(int(r.image_id), r.mask_bbox, tuple(int(S.loc[r.image_id, k]) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")),
                 int(S.loc[r.image_id, "w"]), int(S.loc[r.image_id, "h"])) for r in B.itertuples()]
        with ProcessPoolExecutor(16, mp_context=mp.get_context("spawn")) as ex:
            for iid, win, padded in ex.map(_train_crop, jobs, chunksize=16):
                windows[iid] = win; rec.append(dict(image_id=iid, source="sam_box", padded=padded))
    for r in F.itertuples():
        s = S.loc[r.image_id]
        if r.image_id in windows:
            x0, y0, x1, y1 = windows[r.image_id]
        elif which == "train":
            x0, y0, x1, y1 = int(s.crop_x0), int(s.crop_y0), int(s.crop_x1), int(s.crop_y1)
            (img / f"{r.image_id}.jpg").symlink_to(hk.STAGE / f"{r.image_id}.crop.jpg")
            rec.append(dict(image_id=r.image_id, source="click", padded=0.0))
        else:
            frame = cv2.imread(str(hk.STAGE / f"{r.image_id}.jpg")); H, W = frame.shape[:2]
            L = ((r.head_x - r.tail_x) ** 2 + (r.head_y - r.tail_y) ** 2) ** 0.5
            x0, y0, x1, y1 = hk.square_crop(0.5 * (r.head_x + r.tail_x), 0.5 * (r.head_y + r.tail_y), L, W, H)
            cv2.imwrite(str(img / f"{r.image_id}.jpg"), frame[y0:y1, x0:x1])
        w, h = x1 - x0, y1 - y0
        kx, ky = np.array([r.head_x, r.tail_x]) - x0, np.array([r.head_y, r.tail_y]) - y0
        if not ((kx > 0) & (kx < w) & (ky > 0) & (ky < h)).all():   # the box missed part of the fish: a clipped label is wrong
            (img / f"{r.image_id}.jpg").unlink(missing_ok=True)
            rec.append(dict(image_id=r.image_id, source="dropped: keypoint outside crop", padded=np.nan)); continue
        (lab / f"{r.image_id}.txt").write_text(hk.yolo_label(w, h, r.head_x - x0, r.head_y - y0, r.tail_x - x0, r.tail_y - y0) + "\n")
    msg = f"{which}: {len(F)} crops from {F.dive_id.nunique()} dives"
    if rec:
        R = pd.DataFrame(rec); R.to_csv(hk.HERE / f"results/{sub}_crops.csv", index=False, float_format="%.4f")
        R = R.drop_duplicates("image_id", keep="last"); R.to_csv(hk.HERE / f"results/{sub}_crops.csv", index=False, float_format="%.4f")
        sb = R[R.source == "sam_box"]; nd = int(R.source.str.startswith("dropped").sum())
        msg += f"; {nd} dropped (keypoint outside the crop); {len(sb)} cut by SAM box ({(sb.padded > 0).mean():.1%} padded, median pad {sb.padded[sb.padded > 0].median():.1%}), {len(R) - len(sb) - nd} click-centred"
    print(msg)


def _crop_frame(args):
    """All SAM-box crops of one frame (worker process)."""
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)   # PR_SET_THP_DISABLE: hugepage compaction made big-JPEG decoding crawl here
    cv2.setNumThreads(1)
    image_id, recs = args
    frame = cv2.imread(str(hk.STAGE / f"{image_id}.jpg")); H, W = frame.shape[:2]
    rows = []
    for r in recs:
        x0, y0, x1, y1 = hk.crop_from_box(r["mask_bbox"], W, H)
        name = f"{image_id}_{r['seed']}"
        cv2.imwrite(str(hk.YOLO / "sam" / f"{name}.jpg"), frame[y0:y1, x0:x1])
        rows.append(dict(name=name, image_id=image_id, split=r["split"], seed=r["seed"], crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1,
                         dot_x=r["dot_x"], dot_y=r["dot_y"], sam_head_x=r["head_x"], sam_head_y=r["head_y"], sam_tail_x=r["tail_x"],
                         sam_tail_y=r["tail_y"], sam_status=r["status"]))
    return rows


def sam(workers: int = 16):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    B = pd.DataFrame([json.loads(l) for l in (hk.HERE / "results/sam_boxes.jsonl").open()]).drop_duplicates(["image_id", "seed"], keep="last")
    B = B[B.mask_bbox.notna()]
    out = hk.YOLO / "sam"; shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    jobs = [(int(i), g.to_dict("records")) for i, g in B.groupby("image_id")]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        rows = [r for rs in ex.map(_crop_frame, jobs, chunksize=8) for r in rs]
    C = pd.DataFrame(rows).sort_values(["split", "image_id", "seed"]); C.to_csv(hk.HERE / "results/sam_crops.csv", index=False)
    print("sam crops:", C.groupby(["split", "seed"]).size().to_dict())


def main():
    hk.YOLO.mkdir(parents=True, exist_ok=True)
    for which in sys.argv[1:] or ["train", "val", "sam"]:
        sam() if which == "sam" else train_val(which)
    (hk.YOLO / "data.yaml").write_text(
        f"path: {hk.YOLO}\ntrain: images/train\nval: images/val\nnames:\n  0: fish\nkpt_shape: [2, 3]\n"
        "flip_idx: [0, 1]   # snout and fork are not a left/right pair: a flip keeps their identity\n"
        "# OKS sigmas for the keypoint loss. Ultralytics defaults to 1/n_kpt = 0.5 for a custom skeleton, 5-20x COCO's\n"
        "# (0.025-0.107): a point 10% of the fish off was barely penalised and training plateaued at ~10% snout error.\n"
        "kpt_oks_sigmas: [0.025, 0.025]\n")


if __name__ == "__main__":
    main()
