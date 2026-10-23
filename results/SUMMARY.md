# Snout/fork keypoint model vs the SAM 3.1 rule: length on held-out dives

**Model.** `scripts/kpnet.py`, run `simcc_sam`, epoch 12. It is a top-down SimCC model (ConvNeXt-Tiny,
512-px crop). It was trained on 19,654 human head/tail clicks drawn without a pre-fill, from 117
dives. It was selected on validation-dive length (22 dives) by a rule fixed before any test
prediction (README, Step 3).

**Comparison.** The model and the rule measure the same fish: the crop is cut by the SAM 3.1 mask
box at the laser dot, which is also the mask the rule keypoints. "SAM rule (v1)" is the published
baseline (`e2e_measurement/tail/keypoints.csv`). "SAM rule (v2)" is production's current predictor,
re-run on the same frames; it matches v1 to within 0.4 px on the head and has identical mask
outcomes.

**Intervals.** 95% intervals resample whole dives (2,000 draws). Paired differences use the same
resampled dives for both methods.

**Source.** `cscw-fishsense2027/analysis/codesign/headtail_model.py simcc_sam` writes
`results/headtail_model_*.csv`.

## Reef: 8 calibrated dives, agreement with the human pipeline (not accuracy)

There is no tape on the reef, so the reference is the human pipeline: human laser dot, human
head/tail clicks and stored calibration. d = automatic / human − 1. The frames are those where
every source measured the fish: 265 frames, 119 fish, 7 dives (dive 436 has no such frame).

| Swap | Level | SAM rule (v1): mean \|d\| | Model: mean \|d\| | Paired difference (model − rule) | Within 10%: rule → model |
|---|---|---|---|---|---|
| Head/tail only (human dot, stored calibration) | frame | 5.5% (3.9–6.9) | **3.0% (2.1–3.9)** | **−2.5 pt (−3.2 to −1.8)** | 81% → 94% |
| Head/tail only | fish p90 | 5.3% (3.6–6.9) | **3.1% (2.2–3.8)** | **−2.2 pt (−2.8 to −1.4)** | 86% → 94% |
| Automatic dot + head/tail (stored calibration) | frame | 5.4% (4.1–6.7) | **3.1% (2.0–3.8)** | **−2.3 pt (−3.4 to −1.8)** | 82% → 94% |
| Automatic dot + head/tail | fish p90 | 5.2% (3.8–6.7) | **3.1% (1.8–3.9)** | **−2.1 pt (−3.4 to −1.4)** | 86% → 94% |

Against the v2 rule, the paired differences are −2.6, −2.3, −2.4 and −2.2 points, with intervals
that likewise exclude zero.

The **fully automatic** row, with the size-constancy calibration, was not run. It needs the slate
detector's uncommitted `data/` (the manifest and the SAM masks of slate frames), which this host
does not have.

**Endpoints, head/tail swap, all 265 frames.** Errors are fractions of the human length.

| | Median snout error | Median fork error | Head/tail swapped | Median d |
|---|---|---|---|---|
| SAM rule (v1) | 1.2% | 7.6% | 22.3% | +0.4% |
| Model | 1.1% | **1.2%** | **0.0%** | +0.6% |

**Hogfish** (90 frames, 7 dives): the SAM rule is +4.7% long (median d), with mean \|d\| 5.8%, fork
error 15.3% and 20% swapped. The model is **+0.5%**, with mean \|d\| 2.5%, fork error 1.2% and 0%
swapped. The other species are in `results/headtail_model_reef_species.csv`; every species with 9
or more frames improves on mean \|d\|.

**Largest 10% of disagreements** (27 frames, taken separately for each source). The rule's are
mostly the fork: median fork error 18.5% against snout 1.7%, a median d of −12.6%, and 56% swapped.
The model's are smaller and almost all too long: 93% too long, median d +10.5%, fork 6.0%, snout
3.4%.

## Pool: 10 fish-model dives, tape truth

Score.py's ladder rows use the stored calibration, with per-fish p90 MAE over (dive, fish model):
29 fish. These are the common frames, where every source measured.

| Configuration | Frames | SAM rule (v1): fish p90 MAE | Model: fish p90 MAE | Paired difference (model − rule) | Frame median error: rule → model |
|---|---|---|---|---|---|
| C: human dot, automatic head/tail | 1,413 | **10.2% (8.6–11.4)** | 15.4% (11.4–18.8) | **+5.2 pt (+2.2 to +7.8)** | −8.6% → −6.5% |
| E: automatic dot + head/tail | 1,393 | **10.1% (8.5–11.4)** | 16.4% (11.5–20.6) | **+6.2 pt (+3.0 to +9.3)** | −8.4% → −6.4% |

**Why the pool goes the other way.** This was examined after the result; it was not planned.
Per frame, the model's median is closer to tape for every target. Snook: −10.7% → −7.1%. Grouper:
−3.6% → −0.6%. Purple angel: +7.6% → +0.7%. Shark: +22.6% → −12.2%.

But the model has a heavy upper tail on some targets. The frame p90 for grouper is +14.8% against
−0.8% for the rule; for purple angel it is +40.1% against +10.4%. Production's estimator takes each
fish's 90th-percentile length. That suits the rule, whose lengths are biased short, but for the
model it lands on the over-long frames. Snook, which has the most frames, improves even under p90:
4.4% → 2.8%. Grouper (3.6% → 13.4%) and purple angel (9.5% → 24.7%) get worse.

Only 34 training frames come from the pool, so the plastic models are largely out of domain.

## Validation (22 val dives, 1,739 crops; used for selection only)

| | Mean \|d\| | Median \|d\| | Within 10% | Snout | Fork | Swapped |
|---|---|---|---|---|---|---|
| SAM rule | 6.2% | 2.7% | 79% | 1.3% | 7.7% | 15% |
| Model (epoch 12) | 2.7% | 1.1% | 95% | 1.2% | 1.2% | 0.4% |

## Verdict

- **Reef:** the model beats the SAM rule on length agreement with the human pipeline. Mean \|d\|
  falls from 5.5% to 3.0% for the head/tail swap alone, and from 5.4% to 3.1% with the automatic dot.
  The paired difference is −2.5 points per frame (95% CI −3.2 to −1.8) and −2.2 points per fish.
  Within 10% rises from about 81% to 94%.
  - Nearly all of the gain is at the fork (7.6% → 1.2% of length) and in orientation (22% → 0%
    swapped).
  - The hogfish over-length of +4.7% goes away.
  - This is agreement with the human click convention, not accuracy.
- **Pool:** the model does **not** beat the rule against tape under production's per-fish p90
  estimator. It is worse by 5.2 points (CI +2.2 to +7.8), with the human dot, and by 6.2 points
  with the automatic dot.
  - Its frames are less biased, but a long over-length tail on two fish models moves the p90.
  - The model should not replace the rule for fish models without either more pool training data or
    an estimator matched to its error shape. Either would be a production change and needs sign-off.
- **Limits:**
  - 7 reef dives carry the reef result, and the intervals reflect that.
  - Inter-labeller agreement on real fish is unmeasured: only 12 usable images have two labellers.
    Where labellers differ on fork versus tip, the model can only learn their average convention.
  - The model's remaining reef outliers are fork-end and too long, the same frames that are hard
    for the rule.
