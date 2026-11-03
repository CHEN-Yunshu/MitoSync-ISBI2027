#!/usr/bin/env python
"""Cellpose-SAM segmentation helper (run with the cellres_cp4_blackwell env).

Usage: python cpsam_seg.py --jobs jobs.json --gpu 0
jobs.json = [{"frames_dir": DIR_OF_PNGS, "out": OUT_NPZ}, ...]
Each PNG is converted to 8-bit grayscale (PIL 'L'); masks are saved as npz {"masks": uint16 (T,H,W)}.
Settings (fixed, same as the tracker pipeline): cpsam, diameter=None, flow_threshold=0.4,
cellprob_threshold=0.0, min_size=15, normalize=True, GPU.
"""
import os, sys, json, glob, argparse, time

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
from PIL import Image
from cellpose import models

jobs = json.load(open(args.jobs))
jobs = [j for j in jobs if not (os.path.exists(j["out"]) and not j.get("force", False))]
if not jobs:
    print("[cpsam_seg] nothing to do"); sys.exit(0)
t0 = time.time()
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
print(f"[cpsam_seg] model loaded in {time.time()-t0:.1f}s; {len(jobs)} jobs")
for j in jobs:
    files = sorted(glob.glob(os.path.join(j["frames_dir"], "*.png")))
    frames = [np.asarray(Image.open(f).convert("L")) for f in files]
    t1 = time.time()
    masks = model.eval(frames, batch_size=args.batch, diameter=None, flow_threshold=0.4,
                       cellprob_threshold=0.0, min_size=15, normalize=True, channel_axis=None)[0]
    masks = np.stack([np.asarray(m) for m in masks]).astype(np.uint16)
    os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
    np.savez_compressed(j["out"], masks=masks)
    n = [int(len(np.unique(m)) - 1) for m in masks]
    print(f"[cpsam_seg] {j['frames_dir']} -> {j['out']}  T={len(frames)}  cells/frame={n}  {time.time()-t1:.1f}s")
print(f"[cpsam_seg] done {time.time()-t0:.1f}s")
