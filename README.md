# Head/tail keypoints for FishSense Lite, scored on length

A top-down snout/fork keypoint model (fish box, then a crop with two keypoints) trained on human
head/tail clicks drawn without a pre-fill. It is scored on fish **length** against the SAM 3.1
mask + head/tail rule baseline. Evaluation plugs into the analysis repo (`cscw-fishsense2027`):
`e2e_measurement/score.py` (pool, tape) and `analysis/codesign/field_disagreement.py` (reef,
agreement with the human pipeline).

## Setup

    nix develop          # uv + Python 3.13 + host CUDA libs
    uv sync
    export CSCW_REPO=/path/to/cscw-fishsense2027   # default ../cscw-fishsense2027

## Step 1: data audit (CPU only)

    uv run python scripts/audit.py > results/audit_stdout.txt

This writes `results/audit_*.csv` and the dive-held-out split: `splits/dives.csv` (one row per dive)
and `splits/frames.csv` (one row per eligible image, with the human clicks in label-canvas pixels).

- **Eligible label**: the image has at least one unseeded annotation whose results are all `manual`,
  and **no** seeded or prediction-origin annotation, because `headtail_pixels.psv` keeps only the
  latest label per image. Non-fish targets (Box, checkerboard) are dropped.
- **Test**: the 8 calibrated reef dives plus both sides of the 10 pool pairs (fish-model dives and
  their slate sessions). These dives are never used for training.
- **Validation**: whole reef dives (15% of the remaining reef frames) and whole non-reef dives
  (10%), drawn with seed 0. Checkpoints are selected on the head-tail pixel-length ratio, which
  equals the length ratio at a fixed depth, so the validation dives need no calibration.

`sql/extract_headtail_canvas.sql` gets the label canvas size per annotation. It is still pending,
because it has to run where the restored DB lives.
