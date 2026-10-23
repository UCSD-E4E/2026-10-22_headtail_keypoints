"""Evaluation frames that the label filter leaves out of splits/frames.csv, so they get staged too.

The pool scoring (e2e_measurement/score.py, tape truth) and the tail study's frame list use every
frame of the test dives, including frames whose human head/tail label was pre-filled by SAM 3.1.
Those labels are never used here (not for training, not as truth), but the frames must be staged
so the model and the SAM rule run on the same frames. All are on test dives.

Writes splits/eval_extra.csv (image_id, dive_id, camera_id, split="test", source).
Run: uv run python scripts/eval_frames.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CSCW = Path(os.environ.get("CSCW_REPO", HERE.parent / "cscw-fishsense2027")).resolve()
REEF_TEST = [279, 341, 347, 349, 383, 436, 465, 471]


def main():
    have = set(pd.read_csv(HERE / "splits/frames.csv").image_id)
    test = set(pd.read_csv(HERE / "splits/dives.csv").query("split == 'test'").dive_id)
    T = pd.read_csv(CSCW / "e2e_measurement/tail/frames.csv").drop_duplicates("image_id")
    E = pd.read_csv(CSCW / "e2e_measurement/frames.psv", sep="|").drop_duplicates("image_id")
    parts = [T[(T.set == "reef") & T.dive_id.isin(REEF_TEST) & T.head_x.notna()].assign(source="tail/frames.csv reef"),
             T[T.set == "pool"].assign(source="tail/frames.csv pool"),
             E.assign(source="frames.psv (score.py)")]
    X = pd.concat(p[["image_id", "dive_id", "camera_id", "source"]] for p in parts).drop_duplicates("image_id")
    X = X[~X.image_id.isin(have)].assign(split="test")
    assert X.dive_id.isin(test).all(), "an evaluation frame outside the test dives"
    for c in ("head_x", "head_y", "tail_x", "tail_y"):
        X[c] = float("nan")
    X.sort_values("image_id").to_csv(HERE / "splits/eval_extra.csv", index=False)
    print(X.groupby("source").size().to_string(), f"\ntotal {len(X)}")


if __name__ == "__main__":
    main()
