#!/usr/bin/env python
"""
regen_cond_v2.py -- (re)render condition images of EXISTING clip corpora with cond_render.py version 2 into
<clip>/cond_v2/frame_%03d.png, reading only <clip>/labels/frame_%03d.png + <clip>/lineage.txt.
Never touches labels/, cond/, ref.png, meta.json, target/ or the *_ctc twins.

usage: regen_cond_v2.py --version 2 --workers 8 DIR [DIR ...]      (DIR contains clip_*/ folders)
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cond_render import cond_image, cond_dirname, load_lineage, separation_mask

def do_clip(a):
    cdir, version = a
    lin = load_lineage(os.path.join(cdir, "lineage.txt"))
    out = os.path.join(cdir, cond_dirname(version))
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cdir, "labels", "frame_*.png")))
    sep_px = 0
    for t, f in enumerate(files):
        lab = cv2.imread(f, -1)
        assert lab is not None and lab.ndim == 2, f
        cond = cond_image(lab, lin, t, version)
        if version == 2:
            sep_px += int(separation_mask(lab).sum())
        cv2.imwrite(os.path.join(out, f"frame_{t:03d}.png"), cond[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return os.path.basename(cdir), len(files), sep_px

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--version", type=int, default=2, choices=[1, 2])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for d in args.dirs:
        clips = sorted(glob.glob(os.path.join(d, "clip_*")))
        clips = [c for c in clips if os.path.isdir(os.path.join(c, "labels"))]
        t0 = time.time()
        with Pool(args.workers) as p:
            res = p.map(do_clip, [(c, args.version) for c in clips], chunksize=2)
        nfr = sum(r[1] for r in res); sep = sum(r[2] for r in res)
        info = dict(dir=d, cond_version=args.version, cond_dir=cond_dirname(args.version), n_clips=len(res), n_frames=nfr,
                    separation_px_total=sep, separation_px_per_frame=sep / max(nfr, 1), seconds=time.time() - t0)
        with open(os.path.join(d, f"{cond_dirname(args.version)}_regen.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info), flush=True)

if __name__ == "__main__":
    main()
