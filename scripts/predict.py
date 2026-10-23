"""Run a trained keypoint model on the SAM-box crops of the val and test frames.

For every crop in results/sam_crops.csv (val at the human dot; test at the human and the automatic
dot), the model's snout/fork in frame pixels: the detection containing the laser dot, else the most
confident (hk.pick). Frames where SAM kept no mask have no crop and no model answer: the model is
measured on the fish the production rule would measure, never on a different one.

Writes results/model_keypoints_<name>.csv:
  image_id, split, seed, head_x, head_y, tail_x, tail_y, conf, status (predicted / no_detection),
  and the SAM rule's points on the same crop (sam_head_x ...), for the like-for-like comparison.
Run: uv run python scripts/predict.py --run full [--weights best_length.pt] --gpu 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hk  # noqa: E402
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best_length.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO

    w = hk.DATA / "runs" / a.run / "weights" / a.weights
    C = pd.read_csv(hk.HERE / "results/sam_crops.csv")
    P = T.predict_frames(YOLO(str(w)), C, a.imgsz, device=int(a.gpu))
    P["status"] = P.pred_head_x.notna().map({True: "predicted", False: "no_detection"})
    out = P.rename(columns={f"pred_{c}": c for c in ("head_x", "head_y", "tail_x", "tail_y", "conf")})
    cols = ["image_id", "split", "seed", "head_x", "head_y", "tail_x", "tail_y", "conf", "status",
            "sam_head_x", "sam_head_y", "sam_tail_x", "sam_tail_y", "sam_status", "dot_x", "dot_y"]
    f = hk.HERE / f"results/model_keypoints_{a.run}.csv"
    out[cols].sort_values(["split", "image_id", "seed"]).to_csv(f, index=False, float_format="%.2f")
    print(f"{f.name}: {len(out)} crops; {out.groupby(['split', 'seed', 'status']).size().to_dict()}; weights {w.name}")


if __name__ == "__main__":
    main()
