#!/usr/bin/env python
"""CTC twins for REAL clips ($N/data/vace_clips/<ds>/clip_XXX) -- same conventions as the simulator twins
(lineage_sim.py: uint16 zlib tifs, TRA markers = disc of MARKER_R[ds] at each label centroid restricted to the mask,
one 8-connected component per id, man_track.txt = lineage.txt).

  <clips_root>_ctc/clip_XXX_GT/TRA/man_track%03d.tif + man_track.txt      (arm D: images come from gen_corpus.py)
  <clips_root>_ctc/clip_XXX_ST/SEG/man_seg%03d.tif
  --images: additionally <clips_root>_ctc_real/clip_XXX/t%03d.tif (raw $N/data/ctc/<ds>/02 tifs, cropped as in
            meta.json, native dtype) + the same GT/ST twins next to them  (control arm "real crops").

Real ST labels are not perfect: (a) an id can have >1 connected component in a frame -> keep the largest
4-connected component, the connectivity ctc_io.verify_ctc checks (--no_fix_multi_cc disables); (b) an id can be absent in frames inside [B,E] (temporal gap /
cell leaving and re-entering the crop) -> the gap frame gets the real GT/TRA marker (cropped) if one exists there,
otherwise the track is split into consecutive segments with fresh ids (parent links re-pointed to the last segment).
Every twin is checked with ctc_io.verify_ctc; a summary json goes to <ctc_root>/real_clip_ctc_summary.json.

Usage (cellpose env or vace env):
  python real_clip_ctc.py --dataset PhC-C2DL-PSC [--dataset Fluo-N2DL-HeLa] [--images] [--limit K] [--clips a,b]
"""
import os, sys, json, time, argparse, glob, re
import numpy as np
import cv2
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lineage_sim import markers_image, MARKER_R, S8
from ctc_io import verify_ctc

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", action="append", required=True)
ap.add_argument("--clips_root", default=None, help="default $N/data/vace_clips/<ds>")
ap.add_argument("--ctc_root", default=None, help="default <clips_root>_ctc")
ap.add_argument("--images", action="store_true", help="also write real cropped images + twins to <clips_root>_ctc_real")
ap.add_argument("--real_root", default=None, help="default <clips_root>_ctc_real")
ap.add_argument("--raw_root", default=os.path.join(N, "data/ctc"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--clips", default=None)
ap.add_argument("--no_fix_multi_cc", action="store_true")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), "[real_clip_ctc]", *a, flush=True)


def read_lineage(path):
    rows = np.loadtxt(path, dtype=int, ndmin=2)
    return {int(L): [int(B), int(E), int(P)] for L, B, E, P in rows}


def fix_multi_cc(lab):
    """Keep the largest 4-connected component of every id (verify_ctc uses 4-connectivity). Returns (lab, n_fixed)."""
    n_fixed = 0
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        lid = i + 1
        sub = lab[sl] == lid
        cc, k = ndi.label(sub)  # 4-connectivity
        if k > 1:
            sizes = ndi.sum(sub, cc, index=np.arange(1, k + 1))
            keep = int(np.argmax(sizes)) + 1
            lab[sl][sub & (cc != keep)] = 0
            n_fixed += 1
    return lab, n_fixed


def build_twin(clip_dir, ds, raw_root, fix_cc=True):
    """Returns dict(labels=[uint16], markers=[uint16], lineage={id:[B,E,P]}, stats)."""
    meta = json.load(open(os.path.join(clip_dir, "meta.json")))
    nf = meta["num_frames"]; crop = meta.get("crop"); frames_g = meta["frames_global"]
    lineage = read_lineage(os.path.join(clip_dir, "lineage.txt"))
    labels = []
    n_cc_fixed = 0
    for t in range(nf):
        lab = cv2.imread(os.path.join(clip_dir, "labels", f"frame_{t:03d}.png"), cv2.IMREAD_UNCHANGED)
        assert lab is not None and lab.dtype == np.uint16, f"bad label png {clip_dir} frame {t}"
        lab = lab.astype(np.int32)
        if fix_cc:
            lab, k = fix_multi_cc(lab); n_cc_fixed += k
        labels.append(lab)
    present = {}
    for t, lab in enumerate(labels):
        for lid in np.unique(lab):
            if lid:
                present.setdefault(int(lid), []).append(t)
    assert set(present) == set(lineage), f"label ids != lineage ids in {clip_dir}: {set(present) ^ set(lineage)}"
    rad = MARKER_R[ds]
    markers = [markers_image(lab, rad) for lab in labels]
    # gaps inside [B,E]
    n_gap_frames = n_gap_marker = n_split = 0
    next_id = max(lineage) + 1
    tra_cache = {}
    for lid in sorted(lineage):
        B, E, P = lineage[lid]
        fr = set(present[lid])
        gaps = [t for t in range(B, E + 1) if t not in fr]
        if not gaps:
            continue
        n_gap_frames += len(gaps)
        filled = []
        for t in gaps:
            tg = frames_g[t]
            if tg not in tra_cache:
                tra_cache[tg] = tifffile.imread(os.path.join(raw_root, ds, "02_GT", "TRA", f"man_track{tg:03d}.tif")).astype(np.int32)
            tra = tra_cache[tg]
            if crop:
                tra = tra[crop["y0"]:crop["y0"] + crop["h"], crop["x0"]:crop["x0"] + crop["w"]]
            m = tra == lid
            m &= labels[t] == 0  # never overwrite another instance
            if m.any():
                cc, k = ndi.label(m)  # 4-connectivity
                if k > 1:
                    sizes = ndi.sum(m, cc, index=np.arange(1, k + 1)); m = cc == (int(np.argmax(sizes)) + 1)
                markers[t][m] = lid; filled.append(t); n_gap_marker += 1
        remaining = [t for t in gaps if t not in filled]
        if remaining:
            # split into segments of consecutive presence (marker-filled frames count as present)
            pres = sorted(fr | set(filled))
            segs = []
            for t in pres:
                if segs and t == segs[-1][-1] + 1:
                    segs[-1].append(t)
                else:
                    segs.append([t])
            lineage[lid] = [segs[0][0], segs[0][-1], P]
            last = lid
            for s in segs[1:]:
                nid = next_id; next_id += 1
                for t in s:
                    labels[t][labels[t] == lid] = nid; markers[t][markers[t] == lid] = nid
                lineage[nid] = [s[0], s[-1], 0]
                last = nid; n_split += 1
            if last != lid:
                for c, (cb, ce, cp) in lineage.items():
                    if cp == lid and c != lid and cb > lineage[last][1]:
                        lineage[c][2] = last
    # parent sanity: parent must end before child begins
    for c, (cb, ce, cp) in lineage.items():
        if cp and (cp not in lineage or lineage[cp][1] >= cb):
            lineage[c][2] = 0
    assert max(lineage) < 65536
    stats = dict(n_ids=len(lineage), n_frames=nf, multi_cc_fixed=n_cc_fixed, gap_frames=n_gap_frames,
                 gap_frames_marker_filled=n_gap_marker, tracks_split=n_split, marker_radius=rad, crop=crop)
    return dict(labels=[l.astype(np.uint16) for l in labels], markers=[m.astype(np.uint16) for m in markers],
                lineage=lineage, meta=meta, stats=stats)


def write_twin(root, clip, tw):
    tra = os.path.join(root, f"{clip}_GT", "TRA"); seg = os.path.join(root, f"{clip}_ST", "SEG")
    os.makedirs(tra, exist_ok=True); os.makedirs(seg, exist_ok=True)
    for t, (lab, mk) in enumerate(zip(tw["labels"], tw["markers"])):
        tifffile.imwrite(os.path.join(seg, f"man_seg{t:03d}.tif"), lab, compression="zlib")
        tifffile.imwrite(os.path.join(tra, f"man_track{t:03d}.tif"), mk, compression="zlib")
    with open(os.path.join(tra, "man_track.txt"), "w") as f:
        f.writelines(f"{L} {B} {E} {P}\n" for L, (B, E, P) in sorted(tw["lineage"].items()))


def write_real_images(root, clip, tw, ds, raw_root):
    out = os.path.join(root, clip); os.makedirs(out, exist_ok=True)
    crop = tw["meta"].get("crop"); info = None
    for t, tg in enumerate(tw["meta"]["frames_global"]):
        with tifffile.TiffFile(os.path.join(raw_root, ds, "02", f"t{tg:03d}.tif")) as tf:
            im = tf.pages[0].asarray(); comp = tf.pages[0].compression
        if crop:
            im = im[crop["y0"]:crop["y0"] + crop["h"], crop["x0"]:crop["x0"] + crop["w"]]
        assert im.shape == tw["labels"][t].shape
        tifffile.imwrite(os.path.join(out, f"t{t:03d}.tif"), np.ascontiguousarray(im), compression="lzw" if "LZW" in str(comp) else "zlib")
        info = dict(dtype=str(im.dtype), shape=list(im.shape), src_compression=str(comp))
    return info


def main():
    t0 = time.time()
    for ds in args.dataset:
        clips_root = args.clips_root or os.path.join(N, "data/vace_clips", ds)
        ctc_root = args.ctc_root or clips_root.rstrip("/") + "_ctc"
        real_root = args.real_root or clips_root.rstrip("/") + "_ctc_real"
        os.makedirs(ctc_root, exist_ok=True)
        clips = sorted(os.path.basename(d) for d in glob.glob(os.path.join(clips_root, "clip_*")) if os.path.isdir(d))
        if args.clips:
            want = set(args.clips.split(",")); clips = [c for c in clips if c in want]
        if args.limit:
            clips = clips[:args.limit]
        log(f"{ds}: {len(clips)} clips {clips_root} -> {ctc_root}" + (f" + images -> {real_root}" if args.images else ""))
        results = {}
        agg = dict(n=0, bad_frames=0, missing_ids=0, extra_ids=0, multi_component=0, bad_parents=0, multi_cc_fixed=0,
                   gap_frames=0, gap_frames_marker_filled=0, tracks_split=0, errors=0)
        for c in clips:
            tc = time.time()
            try:
                if not args.force and os.path.exists(os.path.join(ctc_root, f"{c}_GT", "TRA", "man_track.txt")) and \
                        (not args.images or os.path.exists(os.path.join(real_root, c, "t016.tif"))):
                    v = verify_ctc(os.path.join(ctc_root, c)); v.pop("examples", None)
                    results[c] = dict(skipped=True, verify=v)
                else:
                    tw = build_twin(os.path.join(clips_root, c), ds, args.raw_root, fix_cc=not args.no_fix_multi_cc)
                    write_twin(ctc_root, c, tw)
                    r = dict(stats=tw["stats"])
                    if args.images:
                        write_twin(real_root, c, tw)
                        r["real_images"] = write_real_images(real_root, c, tw, ds, args.raw_root)
                        vr = verify_ctc(os.path.join(real_root, c)); r["verify_real_root"] = {k: vr[k] for k in ("bad_frames", "missing_ids", "extra_ids", "multi_component", "bad_parents")}
                    v = verify_ctc(os.path.join(ctc_root, c))
                    r["verify"] = v
                    for k in ("multi_cc_fixed", "gap_frames", "gap_frames_marker_filled", "tracks_split"):
                        agg[k] += tw["stats"][k]
                    results[c] = r
                for k in ("bad_frames", "missing_ids", "extra_ids", "multi_component", "bad_parents"):
                    agg[k] += v[k]
                agg["n"] += 1
                log(f"{c}: ids={v['tracks']} div={v['divisions']} bad_frames={v['bad_frames']} missing={v['missing_ids']} extra={v['extra_ids']} "
                    f"multi={v['multi_component']} bad_parents={v['bad_parents']} "
                    + (f"stats={results[c]['stats']}" if "stats" in results[c] else "(skipped)") + f" {time.time() - tc:.1f}s")
            except Exception as e:
                import traceback
                agg["errors"] += 1; results[c] = dict(error=f"{type(e).__name__}: {e}")
                log(f"ERROR {c}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        summ = dict(dataset=ds, clips_root=clips_root, ctc_root=ctc_root, real_root=real_root if args.images else None,
                    aggregate=agg, clips=results, seconds=round(time.time() - t0, 1))
        with open(os.path.join(ctc_root, "real_clip_ctc_summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
        log(f"{ds} SUMMARY {agg} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
