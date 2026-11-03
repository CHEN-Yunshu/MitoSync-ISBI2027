#!/usr/bin/env python
"""Build a VACE LoRA training dataset (PNG-folder format, lossless) from CTC frames + ST masks.

Dataset layout (consumed by train_lora.py; paths in metadata.csv are relative to the dataset root):
  <root>/metadata.csv            columns: video,vace_video,vace_reference_image,prompt
  <root>/<clip>/video/frame_%03d.png     real frames, RGB 8-bit (gray replicated)   == training target
  <root>/<clip>/control/frame_%03d.png   control frames (R contours, G norm. DT, B 0) == vace_video
  <root>/<clip>/ref.png                  reference image (real first frame, RGB)     == vace_reference_image
  <root>/<clip>/clip_info.json           provenance (dataset/video/start/gap/frames)

Example (4 dummy clips, 17 frames, from PSC 02):
  python make_dummy_dataset.py --out $N/data/vace_dummy --dataset PhC-C2DL-PSC --video 02 \
      --num_clips 4 --num_frames 17 --starts 0,60,120,180 --gap 1
"""
import os, sys, json, argparse, csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vace_common import render_control, gray_to_rgb_pil, ctc_paths, load_ctc_frame, load_ctc_seg

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--ctc_root", default="STUDY_ROOT/data/ctc")
p.add_argument("--dataset", default="PhC-C2DL-PSC")
p.add_argument("--video", default="02", help="CTC video used as source. NEVER 01 (held-out test).")
p.add_argument("--num_clips", type=int, default=4)
p.add_argument("--num_frames", type=int, default=17, help="4k+1")
p.add_argument("--starts", default=None, help="comma list of start frames; default = evenly spaced")
p.add_argument("--gap", type=int, default=1, help="temporal subsampling step between consecutive frames")
p.add_argument("--prompt", default="phase-contrast microscopy time-lapse video of living cells migrating and dividing on a flat substrate, grayscale, sharp, high detail")
p.add_argument("--ref_mode", default="first", choices=["first", "random_other"], help="reference image = first frame of the clip, or a random frame from elsewhere in the same video")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

assert args.video != "01", "video 01 is the held-out evaluation video; never use it for training data"
assert (args.num_frames - 1) % 4 == 0, "num_frames must be 4k+1"
img_dir, seg_dir = ctc_paths(args.ctc_root, args.dataset, args.video)
n_total = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
span = (args.num_frames - 1) * args.gap + 1
if args.starts:
    starts = [int(s) for s in args.starts.split(",")]
else:
    starts = np.linspace(0, n_total - span, args.num_clips).astype(int).tolist()
rng = np.random.default_rng(args.seed)

os.makedirs(args.out, exist_ok=True)
rows = []
for ci, s in enumerate(starts):
    clip = f"clip_{ci:03d}"
    cdir = os.path.join(args.out, clip)
    os.makedirs(os.path.join(cdir, "video"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "control"), exist_ok=True)
    ts = [s + k * args.gap for k in range(args.num_frames)]
    assert ts[-1] < n_total, f"clip {clip} exceeds video length"
    for k, t in enumerate(ts):
        fr = load_ctc_frame(img_dir, t)
        seg = load_ctc_seg(seg_dir, t)
        assert fr.shape == seg.shape
        gray_to_rgb_pil(fr).save(os.path.join(cdir, "video", f"frame_{k:03d}.png"))
        Image.fromarray(render_control(seg), "RGB").save(os.path.join(cdir, "control", f"frame_{k:03d}.png"))
    if args.ref_mode == "first":
        ref_t = ts[0]
    else:
        cand = [t for t in range(n_total) if t < ts[0] - 5 or t > ts[-1] + 5]
        ref_t = int(rng.choice(cand))
    gray_to_rgb_pil(load_ctc_frame(img_dir, ref_t)).save(os.path.join(cdir, "ref.png"))
    h, w = fr.shape
    with open(os.path.join(cdir, "clip_info.json"), "w") as f:
        json.dump(dict(dataset=args.dataset, video=args.video, frames=ts, ref_frame=ref_t, height=h, width=w, gap=args.gap), f)
    rows.append(dict(video=f"{clip}/video", vace_video=f"{clip}/control", vace_reference_image=f"{clip}/ref.png", prompt=args.prompt))
    print(f"{clip}: frames {ts[0]}..{ts[-1]} (gap {args.gap}), ref t{ref_t:03d}, {h}x{w}")

with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["video", "vace_video", "vace_reference_image", "prompt"])
    wr.writeheader()
    wr.writerows(rows)
print(f"wrote {len(rows)} clips -> {os.path.join(args.out, 'metadata.csv')}")
