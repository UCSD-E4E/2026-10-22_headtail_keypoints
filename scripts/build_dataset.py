"""Build the keypoint datasets from the staged frames (CPU).

  train    YOLO-pose images/labels from the staged training crops (<id>.crop.jpg, origin in
           results/stage_log.jsonl). One fish per crop: box from the clicks (hk.box_from_keypoints),
           keypoint 0 = snout (human "head"), 1 = fork (human "tail").
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
import shutil
import sys

import cv2
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import hk  # noqa: E402

LOG = hk.HERE / "results/stage_log.jsonl"


def staged() -> pd.DataFrame:
    L = pd.DataFrame([json.loads(l) for l in LOG.open()])
    L = L[L.get("error").isna()] if "error" in L else L
    return L.drop_duplicates("image_id", keep="last")   # two staging runs overlapped on some frames; same bytes either way


def train_val(which: str):
    F = pd.read_csv(hk.HERE / "splits/frames.csv")
    S = staged().set_index("image_id")
    F = F[(F.split == which) & F.image_id.isin(S.index)]
    img, lab = hk.YOLO / "images" / which, hk.YOLO / "labels" / which
    for d in (img, lab):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    n = 0
    for r in F.itertuples():
        s = S.loc[r.image_id]
        if which == "train":
            x0, y0 = int(s.crop_x0), int(s.crop_y0); w, h = int(s.crop_x1) - x0, int(s.crop_y1) - y0
            (img / f"{r.image_id}.jpg").symlink_to(hk.STAGE / f"{r.image_id}.crop.jpg")
        else:
            frame = cv2.imread(str(hk.STAGE / f"{r.image_id}.jpg")); H, W = frame.shape[:2]
            L = ((r.head_x - r.tail_x) ** 2 + (r.head_y - r.tail_y) ** 2) ** 0.5
            x0, y0, x1, y1 = hk.square_crop(0.5 * (r.head_x + r.tail_x), 0.5 * (r.head_y + r.tail_y), L, W, H)
            w, h = x1 - x0, y1 - y0
            cv2.imwrite(str(img / f"{r.image_id}.jpg"), frame[y0:y1, x0:x1])
        (lab / f"{r.image_id}.txt").write_text(hk.yolo_label(w, h, r.head_x - x0, r.head_y - y0, r.tail_x - x0, r.tail_y - y0) + "\n")
        n += 1
    print(f"{which}: {n} crops from {F.dive_id.nunique()} dives")


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
        "flip_idx: [0, 1]   # snout and fork are not a left/right pair: a flip keeps their identity\n")


if __name__ == "__main__":
    main()
