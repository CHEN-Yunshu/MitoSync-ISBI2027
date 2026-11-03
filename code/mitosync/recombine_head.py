"""[MitoSync copy of scripts/vace/recombine.py: source and output are explicit paths (--src_root/--out) so the head-only
recombination (Ch-head) reads ONLY <ds>_trainhead (02 head) and writes to data/ctc_head/<ds>_recomb; label/marker tifs are
zlib-compressed (disk); everything else unchanged.]

Zero-cost real-trajectory recombination for any CTC dataset (arm C).

A flipped + rolled copy of video <v> is overlaid onto itself wherever the copy cell is free (>= --free_frac of its area
not covered by original cells); copy cells keep contiguous-run track ids, parent links are preserved when the parent's
run ends exactly at the daughter's birth - 1 (so divisions survive if the whole local lineage is pasteable).
  mode full : every frame, every copy cell                        -> data/ctc/<ds>_recomb/<v>
  mode div  : only (+-half_win)-frame windows around real divisions of the copy (parent track + both daughters), i.e. a
              division-rich dataset with ~2x the divisions          -> data/ctc/<ds>_recomb_div/<v>
usage: recombine.py <dataset> <video> [--mode full|div] [--dy DY --dx DX] [--copies K] [--half_win 8] [--free_frac 0.6]
       [--out_suffix NAME]   (default offsets: dy = 0.2 H, dx = 0.35 W; copies k>1 use additional offsets)
Images keep the source dtype/intensity range; pasted cells are feathered and median-harmonised to their local background.
Lineage consistency is verified at the end with ctc_io.verify_ctc (every id present in frames B..E, one component).
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, tifffile, cv2
from scipy import ndimage as ndi
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vace"))  # ctc_io lives in scripts/vace
from ctc_io import D, tn, read_tracks, write_tracks, verify_ctc

ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('video')
ap.add_argument('--mode', choices=['full', 'div'], default='full'); ap.add_argument('--dy', type=int, default=None); ap.add_argument('--dx', type=int, default=None)
ap.add_argument('--copies', type=int, default=1); ap.add_argument('--half_win', type=int, default=8); ap.add_argument('--free_frac', type=float, default=0.6)
ap.add_argument('--out_suffix', default=None); ap.add_argument('--marker', choices=['mask', 'disc'], default='mask')
ap.add_argument('--src_root', required=True, help='CTC dataset root containing <video>, <video>_GT/TRA, <video>_ST/SEG')
ap.add_argument('--out', required=True, help='output dataset root (new path)')
a = ap.parse_args()
src = Path(a.src_root); v = a.video; out = Path(a.out)
assert not (out / v).exists(), f'refusing to overwrite {out / v}'
for d in (out / v, out / f'{v}_GT/TRA', out / f'{v}_ST/SEG'): d.mkdir(parents=True, exist_ok=True)
frames = sorted((src / v).glob('t*.tif')); T = len(frames)
tracks = read_tracks(src / f'{v}_GT/TRA/man_track.txt'); byid = {int(L): (int(B), int(E), int(P)) for L, B, E, P in tracks}
img0 = tifffile.imread(frames[0]); H, W = img0.shape[:2]; dtype = img0.dtype; vmax = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
dy0 = a.dy if a.dy is not None else int(0.2 * H); dx0 = a.dx if a.dx is not None else int(0.35 * W)
offsets = [(dy0, dx0)] + [(int((0.2 + 0.23 * k) * H) % H, int((0.35 + 0.31 * k) * W) % W) for k in range(1, a.copies)]
transforms = [(lambda m, dy=dy, dx=dx: np.roll(np.roll(m[:, ::-1], dy, 0), dx, 1)) for dy, dx in offsets]
maxid = int(tracks[:, 0].max()); id_base = [maxid + 1000 * (c + 1) + maxid * c for c in range(a.copies)]  # disjoint id ranges per copy
t0 = time.time()

# ---- which (copy, label, t) triples are candidates
if a.mode == 'full':
    cand = None  # everything
else:
    cand = set()
    for L, (B, E, P) in byid.items():
        if P == 0: continue
        if B != byid[P][1] + 1: continue  # malformed division
        td = byid[P][1]  # last frame of parent
        for t in range(max(0, td - a.half_win + 1), td + 1): cand.add((P, t))  # parent: half_win frames incl. division frame
        for t in range(B, min(T - 1, B + a.half_win - 1) + 1): cand.add((L, t))  # daughter: half_win frames after
    # make sure both daughters are included (second daughter adds itself in its own iteration)
    print(json.dumps(dict(mode='div', candidate_cell_frames=len(cand), divisions=sum(1 for L in byid if byid[L][2]))), flush=True)

# ---- pass 1: survivors per frame (largest free component >= free_frac of the copy cell), per copy
segs = {}; alive = [{} for _ in range(a.copies)]; keepmask_cache = {}
for fr in frames:
    t = tn(fr); seg = tifffile.imread(src / f'{v}_ST/SEG' / f'man_seg{t:03d}.tif').astype(np.int32)
    tra = tifffile.imread(src / f'{v}_GT/TRA' / f'man_track{t:03d}.tif').astype(np.int32)
    fg = (seg > 0) | (tra > 0)  # original cells = ST masks AND GT markers (a marker-only cell must not be covered)
    segs[t] = seg; segc = []
    for c, tf in enumerate(transforms):
        seg2 = tf(seg); segc.append(seg2)
        for lab in np.unique(seg2):
            lab = int(lab)
            if lab == 0 or (cand is not None and (lab, t) not in cand): continue
            obj = seg2 == lab; free = obj & ~fg
            if free.sum() < a.free_frac * obj.sum(): continue
            cl, n = ndi.label(free)
            if n > 1:
                sizes = ndi.sum(free, cl, range(1, n + 1)); free = cl == (1 + int(np.argmax(sizes)))
                if free.sum() < a.free_frac * obj.sum(): continue
            ys, xs = np.nonzero(free); sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            alive[c].setdefault(lab, []).append(t); keepmask_cache[(c, lab, t)] = (sl, free[sl].copy())
            fg = fg | free  # copies of different offsets must not overlap each other either
    segs[t] = (seg, segc)

# ---- contiguous runs -> new ids, parent links preserved where the local lineage is intact
newid = {}; lines = [(L, B, E, P) for L, B, E, P in tracks]; run_end = {}  # (c, lab, e) -> id of the run ending at e
for c in range(a.copies):
    nxt = id_base[c] + maxid + 1
    runs_by_lab = {}
    for lab, ts in alive[c].items():
        ts = sorted(ts); runs = [[ts[0], ts[0]]]
        for t in ts[1:]:
            if t == runs[-1][1] + 1: runs[-1][1] = t
            else: runs.append([t, t])
        runs_by_lab[lab] = runs
        for i, (b, e) in enumerate(runs):
            lid = lab + id_base[c] if i == 0 else nxt
            if i > 0: nxt += 1
            for t in range(b, e + 1): newid[(c, lab, t)] = lid
            run_end[(c, lab, e)] = lid
    for lab, runs in runs_by_lab.items():
        B, E, P = byid[lab]
        for b, e in runs:
            lid = newid[(c, lab, b)]
            par = run_end.get((c, P, B - 1), 0) if (P and b == B) else 0  # parent copy-run must end exactly at B-1
            lines.append((lid, b, e, par))
from collections import Counter
n_copy = len(lines) - len(tracks); n_copy_dau = sum(1 for L, B, E, P in lines[len(tracks):] if P)
n_copy_div = sum(1 for v in Counter(P for L, B, E, P in lines[len(tracks):] if P).values() if v == 2)  # division events (2 daughters)
print(json.dumps(dict(pass1_seconds=round(time.time() - t0), copy_tracks=n_copy, copy_daughters=n_copy_dau, copy_divisions=n_copy_div, original_tracks=len(tracks),
                      original_daughters=sum(1 for L in byid if byid[L][2]), original_divisions=sum(1 for v in Counter(byid[L][2] for L in byid if byid[L][2]).values() if v == 2))), flush=True)

# ---- pass 2: write images / masks / markers
for fr in frames:
    t = tn(fr); img = tifffile.imread(fr).astype(np.float32); seg, segc = segs[t]
    tra = tifffile.imread(src / f'{v}_GT/TRA' / f'man_track{t:03d}.tif').astype(np.int32)
    new_seg = seg.copy(); new_tra = tra.copy(); new_img = img.copy(); fg = (seg > 0) | (tra > 0); imgc = [tf(img) for tf in transforms]
    for c in range(a.copies):
        for lab in alive[c]:
            if (c, lab, t) not in newid: continue
            sl, km = keepmask_cache[(c, lab, t)]; keep = np.zeros((H, W), bool); keep[sl] = km; obj = segc[c] == lab; img2 = imgc[c]
            ring = ndi.binary_dilation(obj, iterations=3) & ~obj & ~fg
            shift_i = float(np.median(img[ring]) - np.median(img2[ring])) if ring.sum() > 10 else 0.
            alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.0); alpha = np.clip(alpha / max(alpha.max(), 1e-6), 0, 1)
            alpha[~ndi.binary_dilation(keep, iterations=1) | fg] = 0
            new_img = new_img * (1 - alpha) + np.clip(img2 + shift_i, 0, vmax) * alpha
            lid = newid[(c, lab, t)]; new_seg[keep] = lid; fg |= keep
            if a.marker == 'mask': new_tra[keep & (new_tra == 0)] = lid
            else:
                cy, cx = ndi.center_of_mass(keep); rr, cc = np.ogrid[:H, :W]
                disc = (((rr - cy) ** 2 + (cc - cx) ** 2) <= 9) & keep & (new_tra == 0)
                new_tra[disc if disc.any() else (keep & (new_tra == 0))] = lid
    tifffile.imwrite(out / v / f't{t:03d}.tif', np.clip(np.round(new_img), 0, vmax).astype(dtype))
    tifffile.imwrite(out / f'{v}_ST/SEG' / f'man_seg{t:03d}.tif', new_seg.astype(np.uint16), compression='zlib'); tifffile.imwrite(out / f'{v}_GT/TRA' / f'man_track{t:03d}.tif', new_tra.astype(np.uint16), compression='zlib')
write_tracks(out / f'{v}_GT/TRA/man_track.txt', sorted(lines))
summary = dict(out=str(out), mode=a.mode, offsets=offsets, frames=T, original_tracks=len(tracks), copy_tracks=n_copy, copy_daughters=n_copy_dau, copy_divisions=n_copy_div, seconds=round(time.time() - t0))
print(json.dumps(summary), flush=True)
ver_copy = verify_ctc(out / v, min_id=maxid + 1); print('VERIFY_COPIES_ONLY', json.dumps(ver_copy), flush=True)
ver = verify_ctc(out / v); print('VERIFY_ALL', json.dumps(ver), flush=True)
json.dump(dict(summary=summary, verify_copies_only=ver_copy, verify_all=ver), open(out / f'{v}_recombine_summary.json', 'w'), indent=1)
