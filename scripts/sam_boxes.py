"""Production's SAM 3.1 head/tail stage on the staged test and val frames: the baseline, and the fish box.

For every staged test/val frame and each available dot -- "human" (the human laser label,
data/db_extracts/laser_dots.psv) and "auto" (the deployed laser detector's dot, e2e_measurement/
dots.jsonl + tail/reef_dots.jsonl, as tail/evaluate.py loads them; test frames only) -- this calls
fishsense-services' own `predict_from_jpeg` (headtail_predict/predict.py): one 1800x1350 window at
the dot, SAM 3.1 prompted "fish", the mask under the dot (`mask_at_point`), fishsense-core's
`FishHeadTailDetector` on it. It records
  * the SAM-rule head/tail (the baseline, as production computes it today), and
  * the kept mask's box `mask_bbox` -- the fish box the keypoint model crops by, so the model and
    the rule see the same fish chosen the same way (the mask under the laser dot).
The mask itself is saved (bit-packed, cropped to its box) under $HEADTAIL_DATA/sam/ for later checks.

Runs in the processor's own environment, built from a pinned fishsense-services checkout
(see README "SAM environment"); production code is imported, never modified.
Writes results/sam_boxes.jsonl (numbers only). Resumable.
Run: $SAM_PY scripts/sam_boxes.py [--gpu N] [--limit N] [--splits test,val]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CSCW = Path(os.environ.get("CSCW_REPO", HERE.parent / "cscw-fishsense2027")).resolve()
DATA = Path(os.environ.get("HEADTAIL_DATA", f"/local/{os.environ.get('USER', 'user')}/headtail_keypoints"))
STAGE = DATA / "frames"
MASKS = DATA / "sam"
OUT = HERE / "results/sam_boxes.jsonl"
CKPT = Path(os.environ.get("FISHSENSE_SAM3_CHECKPOINT", HERE.parent / "fishsense-detector/models/sam3.1_multiplex.pt"))
CKPT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"   # fishsense-detector weights.toml


def human_dots() -> dict:
    L = pd.read_csv(CSCW / "data/db_extracts/laser_dots.psv", sep="|", header=None, names=["image_id", "x", "y", "label"])
    L = L.dropna(subset=["x", "y"]).drop_duplicates("image_id", keep="last")
    return {int(r.image_id): (float(r.x), float(r.y)) for r in L.itertuples()}


def auto_dots() -> dict:
    """Latest dot without an error, per image (tail/evaluate.py's load())."""
    d = {}
    for name in (CSCW / "e2e_measurement/dots.jsonl", CSCW / "e2e_measurement/tail/reef_dots.jsonl"):
        for l in name.open():
            r = json.loads(l)
            if "dot" in r and r["dot"]["x"] is not None:
                d[int(r["image_id"])] = (float(r["dot"]["x"]), float(r["dot"]["y"]))
    return d


def jobs(splits: list[str]) -> list[tuple[int, str, str, tuple]]:
    F = pd.concat([pd.read_csv(HERE / "splits/frames.csv"), pd.read_csv(HERE / "splits/eval_extra.csv")])
    F = F[F.split.isin(splits)].drop_duplicates("image_id")
    staged = {int(json.loads(l)["image_id"]) for l in (HERE / "results/stage_log.jsonl").open() if "error" not in l}
    done = {(r["image_id"], r["seed"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    H, A = human_dots(), auto_dots()
    out = []
    for r in F.sort_values(["split", "image_id"]).itertuples():
        iid = int(r.image_id)
        if iid not in staged:
            continue
        for seed, pt in (("human", H.get(iid)), ("auto", A.get(iid) if r.split == "test" else None)):
            if pt is not None and (iid, seed) not in done:
                out.append((iid, r.split, seed, pt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="CUDA device index (default: as set in CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="test,val")
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from fishsense_services_processor.headtail_predict.predict import PredictOptions, _Sam3Adapter, get_segmenter, predict_from_jpeg

    todo = jobs(a.splits.split(","))[: a.limit]
    print(f"sam: {len(todo)} (frame, seed) jobs; cuda {torch.cuda.is_available()}", flush=True)
    if not todo:
        return
    sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert sha == CKPT_SHA256, f"SAM 3.1 checkpoint hash mismatch: {sha}"
    seg = _Sam3Adapter(get_segmenter(str(CKPT)))
    opts = PredictOptions(checkpoint=CKPT.name)
    MASKS.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(exist_ok=True)

    class Sink:
        def keep(self, binary, ox, oy, w, h):
            self.m = (binary, ox, oy)

    t0 = time.time()
    with OUT.open("a") as fh:
        for n, (iid, split, seed, (x, y)) in enumerate(todo, 1):
            rec, t1, sink = dict(image_id=iid, split=split, seed=seed, dot_x=x, dot_y=y), time.time(), Sink()
            try:
                res = predict_from_jpeg((STAGE / f"{iid}.jpg").read_bytes(), [(x, y)], seg, uuid.uuid4(), options=opts, mask_sink=sink)
                rec.update({k: v for k, v in res.model_dump().items()
                            if k in ("status", "head_x", "head_y", "tail_x", "tail_y", "mask_area_px", "crop_x", "crop_y", "mask_bbox", "width", "height")})
                if hasattr(sink, "m"):
                    b, ox, oy = sink.m; ys, xs = np.nonzero(b)
                    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    np.savez_compressed(MASKS / f"{iid}_{seed}.npz", bits=np.packbits(b[y0:y1, x0:x1].astype(bool)),
                                        shape=np.array([y1 - y0, x1 - x0]), offset=np.array([ox + x0, oy + y0]))
            except Exception as e:  # one bad frame is logged, not fatal
                rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
            rec["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if n % 100 == 0:
                el = time.time() - t0
                print(f"{n}/{len(todo)}  {el / 60:.0f} min  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
