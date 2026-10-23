"""Step 1 data audit (CPU only): which human head/tail labels can train and test a keypoint model.

Inputs are the committed read-only DB extracts in the analysis repo (cscw-fishsense2027), found at
$CSCW_REPO (default: ../cscw-fishsense2027):
  data/db_extracts/annotations.psv      one row per Label Studio annotation (sql/extract_annotations.sql)
  data/db_extracts/headtail_pixels.psv  latest completed head/tail clicks per image
  data/db_extracts/field_ht.psv         head/tail + dot + species labels per image
  data/db_extracts/dive_environment.psv reef / pool / pier / other per dive
  e2e_measurement/tail/{frames.csv,keypoints.csv,stage.jsonl}, laser_detection_analysis/intrinsics.csv

Label eligibility is per image, and conservative: an image is a training/test label only if it has
at least one unseeded annotation whose results are all drawn by hand (origins == "manual") and NO
annotation that was seeded or carries a prediction origin. headtail_pixels holds one coordinate pair
per image (the latest label), so an image with any seeded annotation could be returning the
anchored coordinates; it is dropped whole.

Split: whole dives. Test = the 8 calibrated reef dives of field_disagreement.py plus both sides of
the 10 pool pairs in score.py's PAIR (fish-model dives and their slate sessions). Validation = whole
dives drawn at random (seed 0) from the remaining reef dives until they hold >= 15% of the remaining
reef frames, and likewise 10% of the remaining non-reef frames. Everything else trains.

Coordinate frame: the SAM 3.1 head point in keypoints.csv was computed on the rectified JPEG
(stage.py: cv2.undistort, same K, same size). If the human clicks were on the raw frame, the
SAM-minus-human offset would follow undistort(p) - p, which reaches ~100-200 px off-centre with
these lenses (k1 ~ -0.12). The regression slope of the observed offset on that predicted shift is
~0 if the clicks are rectified, ~1 if raw.

Writes results/audit_*.csv and splits/dives.csv, splits/frames.csv. Prints a summary.
Run: uv run python scripts/audit.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CSCW = Path(os.environ.get("CSCW_REPO", HERE.parent / "cscw-fishsense2027")).resolve()
DB = CSCW / "data/db_extracts"
TAIL = CSCW / "e2e_measurement/tail"
OUT = HERE / "results"
SPLITS = HERE / "splits"

REEF_TEST = [279, 341, 347, 349, 383, 436, 465, 471]          # analysis/codesign/field_disagreement.py
POOL_PAIR = {58: 71, 59: 77, 60: 65, 61: 80, 66: 83, 76: 63, 84: 62, 87: 87, 94: 94, 114: 114}  # e2e_measurement/score.py
TEST_DIVES = sorted(set(REEF_TEST) | set(POOL_PAIR) | set(POOL_PAIR.values()))
VAL_FRAC = {"reef": 0.15, "other": 0.10}
SEED = 0
NON_FISH = {"Box", "E4E Checkerboard"}   # labelled targets that are not fish

ANN_COLS = ["kind", "ann_id", "image_id", "dive_id", "annotator", "lead_time_s", "created_at", "seeded",
            "laser_colour", "ref_points", "skipped_points", "upside_down", "slate", "origins"]
FHT_COLS = ["image_id", "dive_id", "camera_id", "taken", "hx", "hy", "tx", "ty", "lx", "ly", "label",
            "grouping", "content", "measurable", "angle_cat", "curved_cat", "angle_deg", "top3"]


def species_of(content) -> str:
    """'Fish, Hogfish (Lachnolaimus maximus)' -> 'Hogfish'; 'Fish Model, Snook' -> 'model: Snook'."""
    if not isinstance(content, str) or not content.strip():
        return "(no species label)"
    head, _, rest = content.partition(", ")
    name = rest.split(" (")[0].strip() if rest else head.strip()
    return f"model: {name}" if head.strip() == "Fish Model" else name


def image_table() -> pd.DataFrame:
    A = pd.read_csv(DB / "annotations.psv", sep="|", header=None, names=ANN_COLS, dtype={"origins": str})
    H = A[A.kind == "headtail"].copy()
    H["origins"] = H.origins.fillna("")
    H["seeded"] = H.seeded.eq("t")
    H["manual_unseeded"] = ~H.seeded & H.origins.eq("manual")
    H["anchored"] = H.seeded | H.origins.str.contains("prediction")
    H["empty"] = ~H.seeded & H.origins.eq("")
    per = H.groupby("image_id").agg(dive_id=("dive_id", "first"), n_ann=("ann_id", "size"),
                                    n_manual_unseeded=("manual_unseeded", "sum"), n_anchored=("anchored", "sum"),
                                    n_empty=("empty", "sum"), n_annotators=("annotator", "nunique")).reset_index()
    P = pd.read_csv(DB / "headtail_pixels.psv", sep="|", header=None, names=["image_id", "head_x", "head_y", "tail_x", "tail_y"])
    per = per.merge(P, on="image_id", how="left")
    per["has_pixels"] = per.head_x.notna() & per.tail_x.notna()
    per["len_px"] = np.hypot(per.head_x - per.tail_x, per.head_y - per.tail_y)
    F = pd.read_csv(DB / "field_ht.psv", sep="|", header=None, names=FHT_COLS).drop_duplicates("image_id")
    per = per.merge(F[["image_id", "camera_id", "content", "measurable"]], on="image_id", how="left")
    per["species"] = per.content.map(species_of)
    cam = pd.read_csv(DB / "dives.psv", sep="|", header=None, names=["dive_id", "camera_id", "dive_datetime"]).set_index("dive_id").camera_id
    per["camera_id"] = per.camera_id.fillna(per.dive_id.map(cam))
    E = pd.read_csv(DB / "dive_environment.psv", sep="|", header=None, names=["dive_id", "environment"])
    per = per.merge(E, on="dive_id", how="left")
    # the path heuristic calls some fish-model sessions "reef"; score.py's PAIR dives are pool sessions
    per.loc[per.dive_id.isin(list(POOL_PAIR) + list(POOL_PAIR.values())), "environment"] = "pool"
    per["non_fish"] = per.species.isin(NON_FISH)
    per["eligible"] = (per.n_manual_unseeded > 0) & (per.n_anchored == 0) & per.has_pixels & (per.len_px > 0) & ~per.non_fish
    return per, H


def split(per: pd.DataFrame) -> pd.DataFrame:
    el = per[per.eligible]
    dives = el.groupby("dive_id").agg(environment=("environment", "first"), n_frames=("image_id", "size")).reset_index()
    dives["split"] = np.where(dives.dive_id.isin(TEST_DIVES), "test", "train")
    rng = np.random.default_rng(SEED)
    for group, frac in VAL_FRAC.items():
        pool = dives[(dives.split == "train") & ((dives.environment == "reef") == (group == "reef"))]
        target, got = frac * pool.n_frames.sum(), 0
        for i in rng.permutation(pool.index):
            if got >= target:
                break
            dives.loc[i, "split"] = "val"; got += dives.loc[i, "n_frames"]
    # every held-out test dive is listed, labels or not
    missing = [d for d in TEST_DIVES if d not in set(dives.dive_id)]
    E = per.drop_duplicates("dive_id").set_index("dive_id").environment
    dives = pd.concat([dives, pd.DataFrame(dict(dive_id=missing, environment=[E.get(d, "?") for d in missing],
                                                n_frames=0, split="test"))], ignore_index=True)
    return dives.sort_values("dive_id").reset_index(drop=True)


def frame_check() -> pd.DataFrame:
    """Observed (SAM head on rectified JPEG - human head) vs the shift a raw-frame click would show."""
    F = pd.read_csv(TAIL / "frames.csv").drop_duplicates("image_id")
    K = pd.read_csv(TAIL / "keypoints.csv")
    K = K[(K.seed == "human") & (K.mask_status == "ok")]
    I = pd.read_csv(CSCW / "laser_detection_analysis/intrinsics.csv")
    intr = {int(r.camera_id): (np.array(json.loads(r.camera_matrix)), np.array(json.loads(r.distortion_coefficients)))
            for r in I.itertuples()}
    M = K.merge(F[["image_id", "camera_id", "set", "head_x", "head_y", "tail_x", "tail_y"]], on="image_id", suffixes=("_sam", ""))
    rows = []
    for r in M.itertuples():
        if r.camera_id not in intr:
            continue
        Km, D = intr[r.camera_id]
        p = np.array([[[r.head_x, r.head_y]]], dtype=np.float64)
        u = cv2.undistortPoints(p, Km, D, P=Km)[0, 0]
        rows.append(dict(image_id=r.image_id, set=r.set, len_px=np.hypot(r.head_x - r.tail_x, r.head_y - r.tail_y),
                         obs_dx=r.head_x_sam - r.head_x, obs_dy=r.head_y_sam - r.head_y,
                         pred_dx=u[0] - r.head_x, pred_dy=u[1] - r.head_y))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True); SPLITS.mkdir(exist_ok=True)
    per, H = image_table()
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)

    import subprocess
    rev = subprocess.run(["git", "-C", str(CSCW), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"analysis repo commit: {rev or 'unknown'}")
    print("\n== head/tail annotations by (seeded, origins)")
    print(H.groupby(["seeded", "origins"]).size().rename("annotations").to_string())
    print("\n== images")
    flow = {
        "images with a head/tail annotation": len(per),
        " with >=1 unseeded all-manual annotation": int((per.n_manual_unseeded > 0).sum()),
        "  ... of which also have a seeded/prediction annotation (dropped)": int(((per.n_manual_unseeded > 0) & (per.n_anchored > 0)).sum()),
        "  ... of which have no head/tail pixel row (dropped)": int(((per.n_manual_unseeded > 0) & (per.n_anchored == 0) & ~per.has_pixels).sum()),
        "  ... of which are non-fish targets (Box, checkerboard; dropped)": int(((per.n_manual_unseeded > 0) & (per.n_anchored == 0) & per.has_pixels & per.non_fish).sum()),
        "eligible images (unseeded manual, with coordinates)": int(per.eligible.sum()),
        " eligible images labelled by >1 annotator": int((per.eligible & (per.n_annotators > 1)).sum()),
        "dives with eligible images": int(per[per.eligible].dive_id.nunique()),
        "images with only empty-result annotations": int(((per.n_manual_unseeded == 0) & (per.n_anchored == 0)).sum()),
        "  ... of which nonetheless have pixel coords": int(((per.n_manual_unseeded == 0) & (per.n_anchored == 0) & per.has_pixels).sum()),
    }
    flow = pd.Series(flow, name="n"); print(flow.to_string()); flow.to_csv(OUT / "audit_flow.csv")

    el = per[per.eligible].copy()
    dives = split(per)
    el = el.merge(dives[["dive_id", "split"]], on="dive_id", how="left")

    print("\n== eligible images by environment x split")
    t = pd.crosstab(el.environment, el.split, margins=True); print(t.to_string()); t.to_csv(OUT / "audit_env_split.csv")
    print("\n== dives by environment x split")
    t = pd.crosstab(dives.environment, dives.split, margins=True); print(t.to_string())

    print("\n== test dives (all must be held out)")
    td = dives[dives.split == "test"].copy()
    td["role"] = td.dive_id.map(lambda d: "reef (calibrated)" if d in REEF_TEST else ("pool fish-model dive" if d in POOL_PAIR else "pool slate session"))
    print(td.to_string(index=False))

    sp = el.groupby(["species", "split"]).size().unstack(fill_value=0)
    sp["total"] = sp.sum(axis=1); sp = sp.sort_values("total", ascending=False)
    print("\n== eligible images by species x split"); print(sp.to_string()); sp.to_csv(OUT / "audit_species_split.csv")

    m = el.groupby(["measurable"], dropna=False).size().rename("images")
    print("\n== eligible images by 'measurable' choice"); print(m.to_string())

    q = el.groupby("environment").len_px.describe(percentiles=[.05, .5, .95]).round(0)
    print("\n== human head-tail length in canvas px"); print(q.to_string()); q.to_csv(OUT / "audit_len_px.csv")

    pd_dive = el.groupby(["dive_id", "environment", "split"]).size().rename("images").reset_index()
    pd_dive.to_csv(OUT / "audit_per_dive.csv", index=False)
    dives.to_csv(SPLITS / "dives.csv", index=False)
    el[["image_id", "dive_id", "camera_id", "environment", "species", "split", "head_x", "head_y", "tail_x", "tail_y"]] \
        .sort_values("image_id").to_csv(SPLITS / "frames.csv", index=False)

    # ---- staged locally (analysis-host cache; ids from the stage logs)
    staged = {json.loads(l)["image_id"] for l in (TAIL / "stage.jsonl").open()}
    sstage = CSCW / "e2e_measurement/species/stage.jsonl"
    sp_staged = {json.loads(l)["image_id"] for l in sstage.open()} if sstage.exists() else set()
    s = el.assign(tail_stage=el.image_id.isin(staged), sam_windows=el.image_id.isin(sp_staged))
    st = s.groupby("split")[["tail_stage", "sam_windows"]].sum().astype(int)
    st["eligible"] = s.groupby("split").size()
    print(f"\n== already staged (tail_stage log: {len(staged)} ids; species sam_windows log: {len(sp_staged)} ids)")
    print(st.to_string()); st.to_csv(OUT / "audit_staged.csv")

    # ---- coordinate frame
    fc = frame_check()
    fc.to_csv(OUT / "audit_frame_check.csv", index=False)
    fc["pred"] = np.hypot(fc.pred_dx, fc.pred_dy)
    # observed offset projected on the predicted shift direction, as a fraction of the predicted shift:
    # ~0 if the clicks are in the rectified frame, ~1 if they are in the raw frame
    fc["along"] = (fc.obs_dx * fc.pred_dx + fc.obs_dy * fc.pred_dy) / fc.pred.clip(lower=1e-9) ** 2
    fc["bin"] = pd.cut(fc.pred, [0, 5, 10, 20, 1000])
    b = fc.groupby("bin", observed=True).agg(frames=("along", "size"), median_pred_shift_px=("pred", "median"),
                                             median_along_ratio=("along", "median"),
                                             median_obs_offset_px=("obs_dx", lambda v: np.median(np.hypot(v, fc.loc[v.index, "obs_dy"]))))
    w = fc[fc.pred > 5]
    boot = [np.median(w.along.sample(len(w), replace=True, random_state=k)) for k in range(2000)]
    print(f"\n== coordinate frame: {len(fc)} frames with a SAM head on the rectified JPEG (human-dot mask)")
    print(b.round(2).to_string())
    print(f"frames with predicted shift > 5 px: median along-ratio {w.along.median():.2f} "
          f"(95% bootstrap {np.percentile(boot, 2.5):.2f}..{np.percentile(boot, 97.5):.2f}); 0 = rectified, 1 = raw")
    b.to_csv(OUT / "audit_frame_check_bins.csv")
    print("  (heads sit near the principal point, where raw and rectified differ by a few px: this test has little power)")

    # what is at stake: length change if the eligible clicks were raw-frame and got undistorted
    I = pd.read_csv(CSCW / "laser_detection_analysis/intrinsics.csv")
    intr = {int(r.camera_id): (np.array(json.loads(r.camera_matrix)), np.array(json.loads(r.distortion_coefficients)))
            for r in I.itertuples()}
    rows = []
    for cam, g in el[el.camera_id.notna()].groupby("camera_id"):
        if int(cam) not in intr:
            continue
        Km, D = intr[int(cam)]
        h = cv2.undistortPoints(g[["head_x", "head_y"]].to_numpy(np.float64)[:, None], Km, D, P=Km)[:, 0]
        t = cv2.undistortPoints(g[["tail_x", "tail_y"]].to_numpy(np.float64)[:, None], Km, D, P=Km)[:, 0]
        rows.append(pd.DataFrame(dict(image_id=g.image_id, split=g.split, environment=g.environment,
                                      dlen=np.hypot(*(h - t).T) / g.len_px.to_numpy() - 1)))
    st = pd.concat(rows)
    q = st.groupby("environment").dlen.describe(percentiles=[.05, .5, .95]).mul(100).round(2)
    q["count"] = st.groupby("environment").size()
    print("\n== stakes: % length change if these clicks were raw-frame (undistorted head and tail vs as stored)")
    print(q.to_string()); q.to_csv(OUT / "audit_frame_stakes.csv")
    print(f"  eligible images without a camera id (not covered): {int(el.camera_id.isna().sum())}")

if __name__ == "__main__":
    main()
