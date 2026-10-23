"""Top-down snout/fork keypoint model with a SimCC head (the RTMPose design), in plain PyTorch.

Why this and not YOLO-pose: YOLO-pose's OKS keypoint loss did not fit two precise points on a long
fish. With Ultralytics' default sigma for a custom skeleton (0.5) training plateaued at ~10% of
length snout error (val mean |d| ~22%); with a COCO-like sigma (0.025) the loss gave no gradient for
far-off points and it did not learn orientation (52% swapped). See results/train_{full,sigma}_length_val.csv.

Here every crop is one fish already chosen by SAM 3.1 at the laser dot (hk.crop_from_box), so no
detection is needed. Input: the crop warped to IN x IN. Backbone: timm ConvNeXt-Tiny (ImageNet
pretrained), last stage (stride 32). Head (SimCC): per keypoint, a classification over IN*K bins for x
and for y (1/K px), trained with KL divergence to a Gaussian target (sigma SIGMA bins); decoded as
the softmax-weighted mean in a +-RADIUS-bin window around the argmax. Keypoint 0 = snout (human
"head"), 1 = fork (human "tail"); augmentation (any rotation, flips, scale, shift, colour) keeps their
identity.

Selection: after each epoch, mean |d| (d = predicted / human head-tail pixel length - 1) on the
validation-dive SAM-box crops (train.selection_set, as at deployment); best -> best_length.pt.

  train:    uv run python scripts/kpnet.py train --name simcc --epochs 40 --gpu 3
  predict:  uv run python scripts/kpnet.py predict --name simcc --gpu 3   (-> results/model_keypoints_<name>.csv)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hk  # noqa: E402

IN, K, SIGMA, RADIUS = 512, 2, 4.0, 6       # input side, bins per pixel, target sigma (bins), decode window (bins)
NK = 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255; STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


# ------------------------------------------------------------------ geometry
def affine(h: int, w: int, rot=0.0, scale=1.0, shift=(0.0, 0.0), flip=False) -> np.ndarray:
    """2x3 matrix: crop pixels -> network input pixels. The crop's longer side fills IN at scale 1."""
    s = scale * IN / max(h, w)
    a = math.radians(rot); c, si = math.cos(a) * s, math.sin(a) * s
    M = np.array([[c, -si, 0.0], [si, c, 0.0]])
    if flip:
        M[0, :2] *= -1
    cx, cy = w / 2 + shift[0] * w, h / 2 + shift[1] * h
    M[:, 2] = np.array([IN / 2, IN / 2]) - M[:, :2] @ np.array([cx, cy])
    return M


def warp(img, M):
    return cv2.warpAffine(img, M, (IN, IN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(hk.PAD_VALUE,) * 3)


def to_tensor(img) -> torch.Tensor:
    x = (img[:, :, ::-1].astype(np.float32) - MEAN) / STD            # BGR -> RGB, normalised
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# ------------------------------------------------------------------ data
def _worker_init(_):
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)   # PR_SET_THP_DISABLE (hugepage compaction stalls this host; see stage.py)
    cv2.setNumThreads(1)


class TrainSet(torch.utils.data.Dataset):
    """yolo/images/train + labels/train (keypoints normalised to the crop) with augmentation."""

    def __init__(self, sub="train", augment=True):
        self.sub = sub
        lab = hk.YOLO / "labels" / sub
        self.items = sorted(p.stem for p in lab.iterdir())
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        iid = self.items[i]
        img = cv2.imread(str(hk.YOLO / "images" / self.sub / f"{iid}.jpg")); h, w = img.shape[:2]
        v = np.array(open(hk.YOLO / "labels" / self.sub / f"{iid}.txt").read().split(), float)
        kp = np.array([[v[5] * w, v[6] * h], [v[8] * w, v[9] * h]])
        rng = np.random.default_rng()
        if self.augment:
            M = affine(h, w, rot=rng.uniform(0, 360), scale=rng.uniform(0.8, 1.2), shift=rng.uniform(-0.06, 0.06, 2),
                       flip=rng.random() < 0.5)
        else:
            M = affine(h, w)
        x = warp(img, M)
        if self.augment:
            x = x.astype(np.float32)
            x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                    # brightness/contrast
            x = x * rng.uniform(0.85, 1.15, 3)[None, None, :]                       # colour cast (water)
            if rng.random() < 0.3:
                x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
            x = np.clip(x, 0, 255).astype(np.uint8)
        t = kp @ M[:, :2].T + M[:, 2]                                               # (2, 2) input pixels
        return to_tensor(x), torch.tensor(t, dtype=torch.float32)


# ------------------------------------------------------------------ model
class SimCC(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=pretrained, features_only=True, out_indices=[3])
        C = self.backbone.feature_info.channels()[-1]; S = IN // 32
        self.reduce = nn.Sequential(nn.Conv2d(C, NK * 32, 7, padding=3), nn.GELU())
        self.mlp = nn.Sequential(nn.LayerNorm(32 * S * S), nn.Linear(32 * S * S, 512), nn.GELU())
        self.cls_x = nn.Linear(512, IN * K); self.cls_y = nn.Linear(512, IN * K)

    def forward(self, x):
        f = self.reduce(self.backbone(x)[-1])                                # (B, NK*32, S, S)
        B = f.shape[0]; f = f.reshape(B, NK, -1)                             # per keypoint (reshape: channels_last)
        h = self.mlp(f)
        return self.cls_x(h), self.cls_y(h)                                  # (B, NK, IN*K) each


def simcc_target(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bins = torch.arange(IN * K, device=t.device, dtype=torch.float32)
    gx = torch.exp(-((bins - t[..., 0:1] * K) ** 2) / (2 * SIGMA ** 2)); gy = torch.exp(-((bins - t[..., 1:2] * K) ** 2) / (2 * SIGMA ** 2))
    return gx / gx.sum(-1, keepdim=True).clamp_min(1e-12), gy / gy.sum(-1, keepdim=True).clamp_min(1e-12)


def simcc_loss(lx, ly, t):
    gx, gy = simcc_target(t)
    return (Fn.kl_div(Fn.log_softmax(lx.float(), -1), gx, reduction="batchmean") + Fn.kl_div(Fn.log_softmax(ly.float(), -1), gy, reduction="batchmean")) / NK


def decode(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax-weighted mean within +-RADIUS bins of the argmax -> input pixels; and the peak probability."""
    p = torch.softmax(logits.float(), -1); am = p.argmax(-1, keepdim=True)
    bins = torch.arange(p.shape[-1], device=p.device).view(*([1] * (p.dim() - 1)), -1)
    w = p * ((bins - am).abs() <= RADIUS)
    return (w * bins).sum(-1) / w.sum(-1).clamp_min(1e-12) / K, p.max(-1).values


@torch.no_grad()
def predict_crops(model, paths, device, batch=32):
    """(N, 2, 2) keypoints in crop pixels and (N,) confidence (mean peak probability)."""
    model.eval(); out_k, out_c = [], []
    for j in range(0, len(paths), batch):
        imgs = [cv2.imread(str(p)) for p in paths[j:j + batch]]
        Ms = [affine(*im.shape[:2]) for im in imgs]
        x = torch.stack([to_tensor(warp(im, M)) for im, M in zip(imgs, Ms)]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lx, ly = model(x)
        (px, cx), (py, cy) = decode(lx), decode(ly)
        pts = torch.stack([px, py], -1).cpu().numpy()                          # (b, NK, 2) input pixels
        for M, q in zip(Ms, pts):
            A = np.vstack([M, [0, 0, 1]]); Ai = np.linalg.inv(A)
            out_k.append(q @ Ai[:2, :2].T + Ai[:2, 2])
        out_c.append(((cx + cy) / 2).mean(-1).cpu().numpy())
    return np.array(out_k), np.concatenate(out_c)


def predict_frames(model, C: pd.DataFrame, device) -> pd.DataFrame:
    k, c = predict_crops(model, [hk.YOLO / "sam" / f"{n}.jpg" for n in C.name], device)
    P = C.reset_index(drop=True).copy()
    P["pred_head_x"], P["pred_head_y"] = k[:, 0, 0] + P.crop_x0, k[:, 0, 1] + P.crop_y0
    P["pred_tail_x"], P["pred_tail_y"] = k[:, 1, 0] + P.crop_x0, k[:, 1, 1] + P.crop_y0
    P["pred_conf"] = c
    return P


# ------------------------------------------------------------------ train / predict
def train(a):
    import train as T
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device(f"cuda:{a.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    run = hk.DATA / "runs" / a.name / "weights"; run.mkdir(parents=True, exist_ok=True)
    log = hk.HERE / f"results/train_{a.name}_length_val.csv"
    V = T.selection_set()
    sam_ref = T.length_summary(V, "sam_")
    print(f"selection set: {len(V)} val crops from {V.dive_id.nunique()} dives; SAM rule mean|d| {sam_ref['mean_abs_d']:.4f}", flush=True)
    ds = TrainSet(a.train_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, worker_init_fn=_worker_init,
                                     drop_last=True, persistent_workers=True, pin_memory=True)
    model = SimCC().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl); warm = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=0.999)
    print(f"train ({a.train_dir}): {len(ds)} crops, {len(dl)} steps/epoch, {a.epochs} epochs", flush=True)
    rows, best = [], math.inf
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, t in dl:
            x, t = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), t.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lx, ly = model(x)
            loss = simcc_loss(lx, ly, t)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); ema.update(model); tot += float(loss)
        s = T.length_summary(predict_frames(ema.module, V, dev), "pred_")
        s.update(epoch=ep, loss=tot / len(dl), minutes=(time.time() - t0) / 60)
        rows.append(s); pd.DataFrame(rows).to_csv(log, index=False)
        torch.save({"model": ema.module.state_dict(), "epoch": ep, "length_val": s}, run / "last.pt")
        if s["mean_abs_d"] < best:
            best = s["mean_abs_d"]; torch.save({"model": ema.module.state_dict(), "epoch": ep, "length_val": s}, run / "best_length.pt")
        print(f"[length] epoch {ep}: mean|d| {s['mean_abs_d']:.4f} median {s['median_abs_d']:.4f} within10 {s['within10']:.3f} "
              f"bias {s['mean_d']:+.4f} snout {s['snout_med']:.4f} fork {s['fork_med']:.4f} swap {s['swap_rate']:.3f} "
              f"loss {s['loss']:.4f} {s['minutes']:.1f} min (best {best:.4f}; SAM rule {sam_ref['mean_abs_d']:.4f})", flush=True)


def predict(a):
    dev = torch.device(f"cuda:{a.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    ck = torch.load(hk.DATA / "runs" / a.name / "weights" / a.weights, map_location=dev, weights_only=False)
    model = SimCC(pretrained=False).to(dev); model.load_state_dict(ck["model"])
    C = pd.read_csv(hk.HERE / "results/sam_crops.csv")
    P = predict_frames(model, C, dev).rename(columns={f"pred_{c}": c for c in ("head_x", "head_y", "tail_x", "tail_y", "conf")})
    P["status"] = "predicted"
    cols = ["image_id", "split", "seed", "head_x", "head_y", "tail_x", "tail_y", "conf", "status",
            "sam_head_x", "sam_head_y", "sam_tail_x", "sam_tail_y", "sam_status", "dot_x", "dot_y"]
    f = hk.HERE / f"results/model_keypoints_{a.name}.csv"
    P[cols].sort_values(["split", "image_id", "seed"]).to_csv(f, index=False, float_format="%.2f")
    print(f"{f.name}: {len(P)} crops; weights {a.weights} (epoch {ck['epoch']}, val mean|d| {ck['length_val']['mean_abs_d']:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--weights", default="best_length.pt")
    ap.add_argument("--train-dir", default="train", help="yolo/images/<dir> (train: click-centred crops; train_sam: SAM-box crops)")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
