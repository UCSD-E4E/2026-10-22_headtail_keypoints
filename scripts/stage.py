"""Step 2 staging (CPU + NAS only; the GPU is hidden): every eligible frame decoded and rectified once.

Each frame of splits/frames.csv is read from the NAS, decoded with fishsense-core's DEFAULT decode
(`RawImage(raw)`, i.e. `DecodeConfig()`) and rectified with its camera's intrinsics
(laser_detection_analysis/intrinsics.csv in the analysis repo). That is exactly production's
head/tail stage (fishsense-services headtail_preprocess/activities.py:
`RectifiedImage(RawImage(raw_bytes), intrinsics)`): the JPEG labellers click on and SAM 3.1 reads,
so the clicks are in its pixel frame and the model sees what the SAM rule sees. (Not
`DecodeConfig.production()`, which adds CLAHE; a first staging pass used it by mistake and was
discarded: results/stage_log_clahe_decode_discarded.jsonl.)
Written at OpenCV's default JPEG quality, the encoder production uses. The decode (~15 s per
frame) is the cost, so it is done once:
  test, val  the full-resolution frame (<id>.jpg), so crops can follow the detector's box later
  train      a full-resolution crop around the human clicks (<id>.crop.jpg; square, side
             1.6 x head-tail length + 128 px, clipped to the frame; origin in the log) and the
             frame at quarter scale (<id>.q.jpg) for the box detector. ~1/5 the bytes of full frames.

Order: test, then validation, then training frames round-robin over dives, so a partial run is
a dive-balanced subsample. Resumable: frames already in the log are skipped.

Inputs:  splits/frames.csv; data/private/image_paths.psv (sql/extract_image_paths.sql; private)
Outputs: $HEADTAIL_STAGE (default /local/$USER/headtail_keypoints/frames)/<image_id>.jpg
         results/stage_log.jsonl (image_id, split, seconds, bytes or error; no paths)
Run: uv run python scripts/stage.py [--workers 16] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker: rawpy/BLAS pools otherwise take every core in every worker
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CSCW = Path(os.environ.get("CSCW_REPO", HERE.parent / "cscw-fishsense2027")).resolve()
NAS = Path(os.environ.get("FISHSENSE_NAS", Path.home() / "mnt/fishsense_data/REEF/data"))
STAGE = Path(os.environ.get("HEADTAIL_STAGE", f"/local/{os.environ.get('USER', 'user')}/headtail_keypoints/frames"))
LOG = HERE / "results/stage_log.jsonl"


def intrinsics() -> dict:
    I = pd.read_csv(CSCW / "laser_detection_analysis/intrinsics.csv")
    return {int(r.camera_id): (np.array(json.loads(r.camera_matrix)), np.array(json.loads(r.distortion_coefficients)))
            for r in I.itertuples()}


def order(F: pd.DataFrame) -> pd.DataFrame:
    rank = {"test": 0, "val": 1, "train": 2}
    F = F.assign(_s=F.split.map(rank)).sort_values(["_s", "dive_id", "image_id"])
    F["_k"] = F.groupby(["split", "dive_id"]).cumcount()          # round-robin over dives within a split
    return F.sort_values(["_s", "_k", "dive_id"]).drop(columns=["_s", "_k"])


CROP_SCALE, CROP_PAD = 1.6, 128


def write_jpg(img, out: Path) -> int:
    import cv2
    ok, enc = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    tmp = out.with_name(out.name + ".part"); tmp.write_bytes(enc.tobytes()); tmp.rename(out)
    return len(enc)


def stage_one(image_id: int, split: str, path: str, camera_matrix, dist, clicks) -> dict:
    import cv2
    from fishsense_core.image.decode import DecodeConfig
    from fishsense_core.image.raw_image import RawImage
    from fishsense_core.image.rectified_image import CameraIntrinsics, RectifiedImage

    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)   # PR_SET_THP_DISABLE: with many workers, hugepage compaction made the decode ~5x slower
    cv2.setNumThreads(1)
    rec, t0 = dict(image_id=int(image_id), split=split), time.time()
    try:
        raw = (NAS / path).read_bytes()
        img = RectifiedImage(RawImage(raw, config=DecodeConfig()),
                             CameraIntrinsics(camera_matrix=camera_matrix, distortion_coefficients=dist)).data
        H, W = img.shape[:2]; rec.update(h=int(H), w=int(W))
        if split != "train":
            rec["bytes"] = write_jpg(img, STAGE / f"{image_id}.jpg")
        else:
            hx, hy, tx, ty = clicks
            half = 0.5 * (CROP_SCALE * float(np.hypot(hx - tx, hy - ty)) + CROP_PAD)
            cx, cy = 0.5 * (hx + tx), 0.5 * (hy + ty)
            x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
            x1, y1 = min(W, int(np.ceil(cx + half))), min(H, int(np.ceil(cy + half)))
            b = write_jpg(img[y0:y1, x0:x1], STAGE / f"{image_id}.crop.jpg")
            q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            rec.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1, bytes=b + write_jpg(q, STAGE / f"{image_id}.q.jpg"))
    except Exception as e:  # logged, not fatal: a missing raw is one frame, not the run
        rec["error"] = f"{type(e).__name__}: {e}".replace(str(NAS), "<NAS>")[:300]
        if "<NAS>" in rec["error"] or "/" in rec["error"]:
            rec["error"] = f"{type(e).__name__}"      # never log a path (names)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", default="splits/frames.csv", help="frame list (splits/eval_extra.csv for the extra test frames)")
    a = ap.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(exist_ok=True)
    F = pd.read_csv(HERE / a.frames)
    P = pd.read_csv(HERE / "data/private/image_paths.psv", sep="|", header=None,
                    names=["image_id", "dive_id", "camera_id", "checksum", "dive_path", "path"]).set_index("image_id")
    F = order(F)
    F["path"] = F.image_id.map(P.path); F["camera_id"] = F.image_id.map(P.camera_id).fillna(F.camera_id)
    done = {json.loads(l)["image_id"] for l in LOG.open()} if LOG.exists() else set()
    todo = F[~F.image_id.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    K = intrinsics()
    print(f"stage: {len(todo)} to do of {len(F)} ({len(done)} logged); {a.workers} workers -> {STAGE}", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex, LOG.open("a") as fh:
        futs = []
        for r in todo.itertuples():
            cam = int(r.camera_id) if pd.notna(r.camera_id) else -1
            if not isinstance(r.path, str) or cam not in K:
                fh.write(json.dumps(dict(image_id=int(r.image_id), split=r.split,
                                         error="no path" if not isinstance(r.path, str) else f"no intrinsics for camera {cam}", seconds=0)) + "\n")
                continue
            futs.append(ex.submit(stage_one, int(r.image_id), r.split, r.path, *K[cam], (r.head_x, r.head_y, r.tail_x, r.tail_y)))
        for f in as_completed(futs):
            fh.write(json.dumps(f.result()) + "\n"); fh.flush(); n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print(f"{n}/{len(futs)}  {el / 60:.0f} min  eta {el / n * (len(futs) - n) / 3600:.1f} h", flush=True)
    print(f"done {n} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
