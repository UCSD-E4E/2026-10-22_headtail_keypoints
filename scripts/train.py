"""Train the snout/fork keypoint model (Ultralytics YOLO-pose on fish crops); select by LENGTH on val dives.

Why YOLO-pose: one package, pretrained COCO-pose weights, a crop-level detector that also returns
which fish it means (we take the detection containing the laser dot), and callbacks that let the
checkpoint be chosen on our own metric. If its direct keypoint regression proves too coarse at the
~1%-of-length level, a heatmap model (RTMPose) is the next thing to try.

Selection: after every epoch the just-saved last.pt predicts on the validation-dive crops cut by the
SAM 3.1 mask box at the human dot (results/sam_crops.csv, split "val"), i.e. as at deployment. Per
frame d = predicted / human head-tail pixel length - 1 (= the length ratio at the frame's depth).
The checkpoint with the lowest mean |d| is kept as best_length.pt; every epoch's numbers go to
results/train_<name>_length_val.csv. Ultralytics' own best.pt (pose mAP) is ignored.

Run: uv run python scripts/train.py --name small --epochs 3 --fraction 0.1 --gpu 3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hk  # noqa: E402


def selection_set() -> pd.DataFrame:
    C = pd.read_csv(hk.HERE / "results/sam_crops.csv")
    C = C[(C.split == "val") & (C.seed == "human")]
    F = pd.read_csv(hk.HERE / "splits/frames.csv")[["image_id", "dive_id", "head_x", "head_y", "tail_x", "tail_y"]]
    return C.merge(F, on="image_id")


def predict_frames(model, C: pd.DataFrame, imgsz: int, batch: int = 16, device=None) -> pd.DataFrame:
    """Model keypoints in frame pixels for each crop of C (columns pred_head_x.. , pred_conf; NaN if none)."""
    import torch
    paths = [str(hk.YOLO / "sam" / f"{n}.jpg") for n in C.name]
    results = []
    for j in range(0, len(paths), batch):   # chunked by hand: given the whole list, the predictor warmed up on it as one batch (OOM)
        results += [(r.boxes.cpu(), r.keypoints.cpu()) for r in model.predict(paths[j:j + batch], imgsz=imgsz, batch=batch, conf=0.01,
                                                                                verbose=False, device=device)]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = []
    for i, (boxes, kps) in enumerate(results):
        res = type("R", (), {"boxes": boxes, "keypoints": kps})
        r = C.iloc[i]
        k = hk.pick(res, (r.dot_x - r.crop_x0, r.dot_y - r.crop_y0))
        out.append(dict(pred_head_x=np.nan, pred_head_y=np.nan, pred_tail_x=np.nan, pred_tail_y=np.nan, pred_conf=np.nan) if k is None else
                   dict(pred_head_x=k[0] + r.crop_x0, pred_head_y=k[1] + r.crop_y0, pred_tail_x=k[2] + r.crop_x0, pred_tail_y=k[3] + r.crop_y0, pred_conf=k[4]))
    return pd.concat([C.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def length_summary(P: pd.DataFrame, pre: str) -> dict:
    ok = P[f"{pre}head_x"].notna()
    e = hk.keypoint_errors(P[ok], pre)
    a = np.abs(e["d"])
    return dict(n=int(ok.sum()), missing=int((~ok).sum()), mean_abs_d=float(a.mean()), median_abs_d=float(np.median(a)),
                within10=float((a <= 0.10).mean()), mean_d=float(e["d"].mean()), snout_med=float(np.median(e["snout"])),
                fork_med=float(np.median(e["fork"])), swap_rate=float(e["swapped"].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val", action="store_true", help="also run Ultralytics' own validation (pose mAP) each epoch; selection never uses it")
    a = ap.parse_args()
    from ultralytics import YOLO

    V = selection_set()
    log = hk.HERE / f"results/train_{a.name}_length_val.csv"
    print(f"selection set: {len(V)} val crops from {V.dive_id.nunique()} dives", flush=True)
    sam_ref = length_summary(V, "sam_")
    print("SAM rule on the selection set:", {k: round(v, 4) for k, v in sam_ref.items()}, flush=True)
    rows, best = [], [np.inf]

    def on_model_save(trainer):
        m = YOLO(str(trainer.last))
        s = length_summary(predict_frames(m, V, a.imgsz, device=int(a.gpu)), "pred_")
        s["epoch"] = trainer.epoch + 1
        rows.append(s); pd.DataFrame(rows).to_csv(log, index=False)
        if s["mean_abs_d"] < best[0]:
            best[0] = s["mean_abs_d"]; shutil.copy(trainer.last, Path(trainer.wdir) / "best_length.pt")
        print(f"[length] epoch {s['epoch']}: mean|d| {s['mean_abs_d']:.4f} median {s['median_abs_d']:.4f} within10 {s['within10']:.3f} "
              f"bias {s['mean_d']:+.4f} snout {s['snout_med']:.4f} fork {s['fork_med']:.4f} swap {s['swap_rate']:.3f} "
              f"(best {best[0]:.4f}; SAM rule {sam_ref['mean_abs_d']:.4f})", flush=True)

    model = YOLO(a.model)
    model.add_callback("on_model_save", on_model_save)
    model.train(data=str(hk.YOLO / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=int(a.gpu),
                project=str(hk.DATA / "runs"), name=a.name, exist_ok=True, fraction=a.fraction, seed=0, deterministic=True,
                mosaic=0.0, close_mosaic=0, fliplr=0.5, flipud=0.5, degrees=30, scale=0.3, translate=0.1,
                cos_lr=True, workers=a.workers, plots=False, val=a.val)


if __name__ == "__main__":
    main()
