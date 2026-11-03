#!/usr/bin/env python
"""
make_clips.py -- build real training clips for the VACE cell-video generator from
official CTC training videos (video 02 ONLY; 01 is the held-out evaluation video).

Usage (vace venv):
  $N/ext/vace_venv/bin/python $N/scripts/vace/make_clips.py \
      --dataset PhC-C2DL-PSC --video 02 --num_frames 17 --stride 4 \
      --out $N/data/vace_clips/PhC-C2DL-PSC

Per clip (clip_XXX, XXX = 0-based clip index; start frame s = XXX*stride):
  target/frame_%03d.png   8-bit gray replicated to RGB (fixed per-dataset percentile mapping)
  cond/frame_%03d.png     RGB lossless condition image (--cond_version 1, default; cond_v2/ for --cond_version 2,
                          see cond_render.py; existing cond/ is never overwritten by version 2):
                            R = 1-px inner contour of every instance (255)
                            G = per-instance normalised EDT (0..255, 255 at instance centre)
                            B = event markers: mother in its last 3 frames before division = 255,
                                daughter in its birth frame = 128, else 0
  ref.png                 real first frame (== target/frame_000.png)
  labels/frame_%03d.png   uint16 ST labels (== track ids), cropped identically
  lineage.txt             clip-local man_track (L B E P), frames re-indexed 0..num_frames-1,
                          P kept only when the parent is present in the clip
  meta.json               dataset, start frame, crop offsets, percentiles, stats, violations
  target.mp4 / cond.mp4   libx264 crf 10 (target yuv420p, cond yuv444p). The PNG folders are
                          the LOSSLESS source of truth; mp4 are a convenience for the loader.
Dataset root:
  intensity_mapping.json  the ONE fixed 8-bit mapping (1st / 99.9th percentile over all frames
                          of video 02); generated images must be mapped back with the same one.
  metadata.csv            video,vace_video,vace_reference_image,prompt (relative paths)
  summary.json / summary.txt

Crops (fixed, deterministic): PSC 576x720 full frame, DIC-HeLa 512x512 full frame,
Fluo-N2DL-HeLa 700x1100 -> 512x768 crops at y0 = (700-512)//2 = 94 and
x0 in {0 (left), (1100-768)//2 = 166 (centre), 1100-768 = 332 (right)}.
Every clip of a crop dataset exists in all 3 crop positions (clip_XXX_{left,centre,right}).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

PROMPTS = {
    "PhC-C2DL-PSC": "time-lapse microscopy video of cells, phase contrast",
    "Fluo-N2DL-HeLa": "time-lapse microscopy video of cells, fluorescence nuclei",
    "DIC-C2DH-HeLa": "time-lapse microscopy video of cells, DIC",
    "Lead-BF": "time-lapse microscopy video of cells, brightfield",
}

# (y0, x0, h, w, name) ; None -> full frame
CROPS = {
    "PhC-C2DL-PSC": [None],
    "DIC-C2DH-HeLa": [None],
    "Fluo-N2DL-HeLa": [(94, 0, 512, 768, "left"), (94, 166, 512, 768, "centre"), (94, 332, 512, 768, "right")],
    # Lead-BF: 2048x2448 brightfield frames, 3x3 grid of native-resolution 576x720 crops (== lineage_sim.REF_CROPS)
    "Lead-BF": [(y0, x0, 576, 720, f"r{i}c{j}") for i, y0 in enumerate((0, 736, 1472)) for j, x0 in enumerate((0, 864, 1728))],
}

P_LOW, P_HIGH = 1.0, 99.9


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- intensity mapping
def exact_percentiles_hist(files, p_low, p_high):
    """Exact (nearest-rank) percentiles over ALL pixels of all frames via an integer histogram."""
    hist = None
    dtype = None
    for f in files:
        im = tifffile.imread(f)
        if dtype is None:
            dtype = im.dtype
            nb = int(np.iinfo(dtype).max) + 1
            hist = np.zeros(nb, dtype=np.int64)
        hist += np.bincount(im.ravel().astype(np.int64), minlength=hist.size)
    cdf = np.cumsum(hist)
    n = int(cdf[-1])

    def pct(p):
        k = max(1, int(np.ceil(p / 100.0 * n)))
        return int(np.searchsorted(cdf, k))

    return pct(p_low), pct(p_high), str(dtype), n, hist


def to_uint8(im, lo, hi):
    x = (im.astype(np.float32) - lo) / float(max(hi - lo, 1))
    return np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


# ----------------------------------------------------------------------------- lineage
def load_man_track(path):
    arr = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return {int(L): (int(B), int(E), int(P)) for L, B, E, P in arr}


def division_events(tracks):
    """Return mothers {mid: last_frame} with >=2 children and daughters {did: birth_frame}."""
    children = {}
    for L, (B, E, P) in tracks.items():
        if P > 0:
            children.setdefault(P, []).append(L)
    mothers, daughters = {}, {}
    for P, ch in children.items():
        if len(ch) >= 2 and P in tracks:
            mothers[P] = tracks[P][1]
            for d in ch:
                daughters[d] = tracks[d][0]
    return mothers, daughters


# ----------------------------------------------------------------------------- condition image
# rendered by the shared cond_render.py (version 1 = original encoding, byte-identical; version 2 = cond_v2/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cond_render import cond_image as render_cond, cond_dirname  # noqa: E402


# ----------------------------------------------------------------------------- video writing
def write_mp4(path, frames, fps, pix_fmt):
    import imageio

    w = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        pixelformat=pix_fmt,
        ffmpeg_params=["-crf", "10", "-preset", "medium"],
    )
    for f in frames:
        w.append_data(f)
    w.close()


# ----------------------------------------------------------------------------- per-clip worker
def process_clip(job):
    (ds, img_dir, seg_dir, tracks, mothers, daughters, s, nf, crop, out_dir, lo, hi, fps, name, _stride, cond_version) = job
    cdir = os.path.join(out_dir, name)
    cond_dir = cond_dirname(cond_version)
    for sub in ("target", cond_dir, "labels"):
        os.makedirs(os.path.join(cdir, sub), exist_ok=True)

    frames_g = list(range(s, s + nf))
    tgt_frames, cond_frames = [], []
    present = {}  # id -> list of local frames present
    violations_cc = 0  # (id, frame) with != 1 connected component
    violations_label = 0  # label present but not a track id / outside its lifetime
    id_frame_cc = []
    for tl, tg in enumerate(frames_g):
        im = tifffile.imread(os.path.join(img_dir, f"t{tg:03d}.tif"))
        lab = tifffile.imread(os.path.join(seg_dir, f"man_seg{tg:03d}.tif")).astype(np.uint16)
        if crop is not None:
            y0, x0, h, w = crop[:4]
            im = im[y0:y0 + h, x0:x0 + w]
            lab = lab[y0:y0 + h, x0:x0 + w]
        im8 = to_uint8(im, lo, hi)
        tgt = np.stack([im8] * 3, -1)
        cond = render_cond(lab, tracks, tg, cond_version, mothers=mothers, daughters=daughters)
        cv2.imwrite(os.path.join(cdir, "target", f"frame_{tl:03d}.png"), tgt[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(cdir, cond_dir, f"frame_{tl:03d}.png"), cond[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(cdir, "labels", f"frame_{tl:03d}.png"), lab, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if tl == 0:
            cv2.imwrite(os.path.join(cdir, "ref.png"), tgt[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tgt_frames.append(tgt)
        cond_frames.append(cond)
        # verification
        ids = np.unique(lab)
        ids = ids[ids > 0]
        for lid in ids:
            lid = int(lid)
            present.setdefault(lid, []).append(tl)
            if lid not in tracks or not (tracks[lid][0] <= tg <= tracks[lid][1]):
                violations_label += 1
            ncc = ndi.label(lab == lid, structure=np.ones((3, 3)))[1]
            if ncc != 1:
                violations_cc += 1
                id_frame_cc.append((lid, tl, int(ncc)))

    # clip-local lineage
    lineage = []
    gaps = 0
    gap_ids = []
    for lid in sorted(present):
        fr = present[lid]
        B, E = fr[0], fr[-1]
        if E - B + 1 != len(fr):
            gaps += 1
            gap_ids.append({"id": lid, "frames_present": fr, "global_BE": list(tracks.get(lid, (None, None, None))[:2])})
        P = tracks[lid][2] if lid in tracks else 0
        if P > 0 and P not in present:
            P = 0
        lineage.append((lid, B, E, P))
    with open(os.path.join(cdir, "lineage.txt"), "w") as f:
        for L, B, E, P in lineage:
            f.write(f"{L} {B} {E} {P}\n")

    # divisions fully inside the clip (mother present in her last frame, >=2 daughters present at birth)
    div_in_clip = 0
    for mid, last in mothers.items():
        if mid in present and last in [s + t for t in present[mid]] and s <= last < s + nf - 1:
            nd = sum(1 for d, b in daughters.items() if tracks[d][2] == mid and b == last + 1 and d in present and (b - s) in present[d])
            if nd >= 2:
                div_in_clip += 1
    # event markers painted (any frame in clip)
    n_mother_marked = sum(1 for mid, last in mothers.items() if mid in present and any(last - 2 <= s + t <= last for t in present[mid]))
    n_daughter_marked = sum(1 for d, b in daughters.items() if d in present and (b - s) in present[d])

    write_mp4(os.path.join(cdir, "target.mp4"), tgt_frames, fps, "yuv420p")
    write_mp4(os.path.join(cdir, cond_dir + ".mp4"), cond_frames, fps, "yuv444p")

    meta = {
        "dataset": ds,
        "video": os.path.basename(img_dir),
        "clip": name,
        "cond_version": int(cond_version),
        "cond_dir": cond_dir,
        "start_frame": s,
        "num_frames": nf,
        "frames_global": frames_g,
        "crop": None if crop is None else {"y0": crop[0], "x0": crop[1], "h": crop[2], "w": crop[3], "name": crop[4]},
        "height": int(tgt_frames[0].shape[0]),
        "width": int(tgt_frames[0].shape[1]),
        "intensity_percentiles": {"p_low": P_LOW, "p_high": P_HIGH, "lo": lo, "hi": hi},
        "n_ids": len(present),
        "mean_cells_per_frame": float(np.mean([len([1 for lid in present if t in present[lid]]) for t in range(nf)])),
        "divisions_in_clip": div_in_clip,
        "mothers_marked": n_mother_marked,
        "daughters_marked": n_daughter_marked,
        "violations_one_cc": violations_cc,
        "violations_one_cc_list": id_frame_cc[:50],
        "violations_label_not_track": violations_label,
        "ids_with_temporal_gaps": gaps,
        "ids_with_temporal_gaps_list": gap_ids[:50],
        "mp4": {"codec": "libx264", "crf": 10, "target_pix_fmt": "yuv420p", "cond_pix_fmt": "yuv444p", "fps": fps,
                "note": "PNG folders are the lossless source of truth"},
    }
    with open(os.path.join(cdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ----------------------------------------------------------------------------- montage
def montage(out_dir, metas, fig_path, nf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metas = sorted(metas, key=lambda m: m["clip"])
    # prefer clips with divisions, spread across the video
    with_div = [m for m in metas if m["divisions_in_clip"] > 0]
    pool = with_div if len(with_div) >= 3 else metas
    pick = [pool[0], pool[len(pool) // 2], pool[-1]] if len(pool) >= 3 else pool
    fr = [0, nf // 2, nf - 1]
    rows = 3 * len(pick)
    cols = len(fr)
    h, w = pick[0]["height"], pick[0]["width"]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2 * h / w), squeeze=False)
    rng = np.random.RandomState(0)
    lut = rng.rand(65536, 3)
    lut[0] = 0
    for ci, m in enumerate(pick):
        cdir = os.path.join(out_dir, m["clip"])
        for fi, t in enumerate(fr):
            tgt = cv2.imread(os.path.join(cdir, "target", f"frame_{t:03d}.png"))[..., ::-1]
            cond = cv2.imread(os.path.join(cdir, m.get("cond_dir", "cond"), f"frame_{t:03d}.png"))[..., ::-1]
            lab = cv2.imread(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), cv2.IMREAD_UNCHANGED)
            for ri, (img, title) in enumerate([(tgt, "target"), (cond, "cond (R contour, G edt, B event)"), (lut[lab], "labels")]):
                ax = axes[3 * ci + ri, fi]
                ax.imshow(img, interpolation="nearest")
                ax.set_axis_off()
                if fi == 0:
                    ax.set_title(f"{m['clip']} s={m['start_frame']} div={m['divisions_in_clip']}\n{title} f{t}", fontsize=8, loc="left")
                else:
                    ax.set_title(f"{title} f{t}", fontsize=8, loc="left")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PROMPTS))
    ap.add_argument("--video", default="02")
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="STUDY_ROOT/data/ctc")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fig", default=None, help="montage png path")
    ap.add_argument("--max_clips", type=int, default=0, help="debug: limit number of clip starts")
    ap.add_argument("--cond_version", type=int, default=1, choices=[1, 2],
                    help="condition encoding (cond_render.py): 1 = original -> cond/, 2 -> cond_v2/")
    args = ap.parse_args()

    if args.video == "01":
        sys.exit("REFUSING: video 01 is the held-out evaluation video and must never be used for training.")
    assert (args.num_frames - 1) % 4 == 0, "num_frames must be 4k+1"

    ds = args.dataset
    img_dir = os.path.join(args.root, ds, args.video)
    seg_dir = os.path.join(args.root, ds, f"{args.video}_ST", "SEG")
    tra_txt = os.path.join(args.root, ds, f"{args.video}_GT", "TRA", "man_track.txt")
    files = sorted(glob.glob(os.path.join(img_dir, "t*.tif")))
    T = len(files)
    log(f"[{ds}] video {args.video}: {T} frames in {img_dir}")
    os.makedirs(args.out, exist_ok=True)

    # 1) ONE fixed intensity mapping over ALL frames of the video
    t0 = time.time()
    lo, hi, dtype, npx, hist = exact_percentiles_hist(files, P_LOW, P_HIGH)
    raw_min = int(np.nonzero(hist)[0][0])
    raw_max = int(np.nonzero(hist)[0][-1])
    frac_sat_hi = float(hist[hi + 1:].sum() / npx)
    frac_sat_lo = float(hist[:lo].sum() / npx)
    mapping = {
        "dataset": ds, "video": args.video, "source_dtype": dtype, "n_frames": T, "n_pixels": npx,
        "p_low": P_LOW, "p_high": P_HIGH, "lo": lo, "hi": hi, "raw_min": raw_min, "raw_max": raw_max,
        "frac_pixels_clipped_below_lo": frac_sat_lo, "frac_pixels_clipped_above_hi": frac_sat_hi,
        "forward": "u8 = round(clip((x - lo) / (hi - lo), 0, 1) * 255)",
        "inverse": "x = u8 / 255 * (hi - lo) + lo   (apply to generated frames to get source-dtype intensities)",
        "note": "fixed per-dataset mapping computed over ALL frames of video 02; never per-frame normalisation",
    }
    with open(os.path.join(args.out, "intensity_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=1)
    log(f"  intensity mapping: dtype={dtype} raw[{raw_min},{raw_max}] lo(p{P_LOW})={lo} hi(p{P_HIGH})={hi} "
        f"clipped below={frac_sat_lo:.5f} above={frac_sat_hi:.5f}  ({time.time()-t0:.1f}s)")

    # 2) lineage
    tracks = load_man_track(tra_txt)
    mothers, daughters = division_events(tracks)
    log(f"  tracks={len(tracks)} divisions(global, >=2 children)={len(mothers)} single-child parent links="
        f"{sum(1 for L,(B,E,P) in tracks.items() if P>0) - len(daughters)}")

    # 3) jobs
    starts = list(range(0, T - args.num_frames + 1, args.stride))
    if args.max_clips:
        starts = starts[: args.max_clips]
    jobs = []
    for s in starts:
        for crop in CROPS[ds]:
            name = f"clip_{s//args.stride:03d}" + ("" if crop is None else f"_{crop[4]}")
            jobs.append((ds, img_dir, seg_dir, tracks, mothers, daughters, s, args.num_frames, crop, args.out, lo, hi, args.fps, name, args.stride, args.cond_version))
    log(f"  {len(starts)} starts x {len(CROPS[ds])} crops = {len(jobs)} clips; workers={args.workers}")

    t0 = time.time()
    with Pool(args.workers) as pool:
        metas = pool.map(process_clip, jobs, chunksize=1)
    log(f"  clips done in {time.time()-t0:.1f}s")

    # 4) metadata.csv (relative paths)
    metas = sorted(metas, key=lambda m: m["clip"])
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "vace_video", "vace_reference_image", "prompt"])
        for m in metas:
            wr.writerow([f"{m['clip']}/target", f"{m['clip']}/{m['cond_dir']}", f"{m['clip']}/ref.png", PROMPTS[ds]])  # PNG folders (lossless); train_lora.py reads folders. mp4 copy kept in metadata_mp4.csv (verifier fix)
    # preview-only variant pointing at the lossy mp4s (official loader route; cond max abs error ~69 -> NOT for training)
    with open(os.path.join(args.out, "metadata_mp4.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "vace_video", "vace_reference_image", "prompt"])
        for m in metas:
            wr.writerow([f"{m['clip']}/target.mp4", f"{m['clip']}/{m['cond_dir']}.mp4", f"{m['clip']}/ref.png", PROMPTS[ds]])

    # 5) summary
    du = 0
    for dp, dn, fn in os.walk(args.out):
        for x in fn:
            du += os.path.getsize(os.path.join(dp, x))
    summary = {
        "dataset": ds, "video": args.video, "num_frames": args.num_frames, "stride": args.stride,
        "n_clips": len(metas), "n_starts": len(starts), "n_crops": len(CROPS[ds]),
        "cond_version": args.cond_version, "cond_dir": cond_dirname(args.cond_version),
        "crops": [None if c is None else {"y0": c[0], "x0": c[1], "h": c[2], "w": c[3], "name": c[4]} for c in CROPS[ds]],
        "clip_hw": [metas[0]["height"], metas[0]["width"]],
        "intensity_mapping": mapping,
        "global_tracks": len(tracks), "global_divisions": len(mothers),
        "divisions_in_clips_total": sum(m["divisions_in_clip"] for m in metas),
        "clips_with_division": sum(1 for m in metas if m["divisions_in_clip"] > 0),
        "mean_ids_per_clip": float(np.mean([m["n_ids"] for m in metas])),
        "mean_cells_per_frame": float(np.mean([m["mean_cells_per_frame"] for m in metas])),
        "violations_one_cc_total": sum(m["violations_one_cc"] for m in metas),
        "violations_label_not_track_total": sum(m["violations_label_not_track"] for m in metas),
        "ids_with_temporal_gaps_total": sum(m["ids_with_temporal_gaps"] for m in metas),
        "disk_bytes": du, "disk_GB": du / 1e9,
        "prompt": PROMPTS[ds],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    lines = []
    lines.append(f"{'clip':<18}{'start':>6}{'crop':>8}{'HxW':>10}{'ids':>5}{'cells/f':>8}{'div':>4}{'momM':>5}{'dauM':>5}{'cc!=1':>6}{'lbl!':>5}{'gaps':>5}")
    for m in metas:
        lines.append(f"{m['clip']:<18}{m['start_frame']:>6}{(m['crop']['name'] if m['crop'] else 'full'):>8}"
                     f"{str(m['height'])+'x'+str(m['width']):>10}{m['n_ids']:>5}{m['mean_cells_per_frame']:>8.1f}{m['divisions_in_clip']:>4}"
                     f"{m['mothers_marked']:>5}{m['daughters_marked']:>5}{m['violations_one_cc']:>6}{m['violations_label_not_track']:>5}{m['ids_with_temporal_gaps']:>5}")
    lines.append("")
    lines.append(f"TOTAL clips={summary['n_clips']} ({summary['n_starts']} starts x {summary['n_crops']} crops) size={summary['clip_hw']} "
                 f"divisions_in_clips={summary['divisions_in_clips_total']} clips_with_div={summary['clips_with_division']} "
                 f"violations_one_cc={summary['violations_one_cc_total']} label_not_track={summary['violations_label_not_track_total']} "
                 f"ids_with_gaps={summary['ids_with_temporal_gaps_total']} disk={summary['disk_GB']:.2f} GB")
    txt = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    log(txt)

    # 6) montage
    if args.fig:
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        montage(args.out, metas, args.fig, args.num_frames)
        log(f"  montage -> {args.fig}")


if __name__ == "__main__":
    main()
