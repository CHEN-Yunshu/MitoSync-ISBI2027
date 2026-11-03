"""MitoSync Task A, steps 1-2: head-only CTC trees and the 8 Delta-8 phase views of the 02 head.

usage: build_head_views.py <dataset> [--gap 8]
  1. verifies <ds>_trainhead/02 == frames 0..cut-1 of <ds>/02 (images, ST, GT markers byte/label-identical, man_track only
     head frames) and <ds>_valtail/02 == frames cut..n-1, then creates
     $N/data/ctc_head/<ds>/{02,02_GT,02_ST} as symlinks to the _trainhead dirs (nothing is copied).
  2. for k in 0..gap-1 writes $N/data/ctc_head/<ds>_d<gap>o<k>/02{,_GT/TRA,_ST/SEG}: frames k, k+gap, ... of the HEAD,
     lineage remapped exactly as scripts/make_ctc_subsampled.py (B' = ceil((B-k)/gap), E' = floor((E-k)/gap), track kept iff
     B' <= E', parent kept iff the parent track survives, else 0). Image / ST files are symlinks to the head frames; the
     GT marker frame is a symlink when the label filter is a no-op (always true for valid CTC GT), else a zlib tif.
Writes $N/data/ctc_head/<ds>_head_views.json (provenance + checks). Never touches existing data.
"""
import sys, re, json, hashlib, os, argparse
from pathlib import Path
import numpy as np, tifffile

N = Path(os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace"))); D = N / 'data/ctc'; OUT = N / 'data/ctc_head'
tn = lambda p: int(re.search(r'(\d+)', Path(p).stem).group(1))
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('--gap', type=int, default=8); a = ap.parse_args()
ds, gap = a.ds, a.gap
full, head, tail = D / ds, D / f'{ds}_trainhead', D / f'{ds}_valtail'
h = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
rd = lambda p: np.loadtxt(p, dtype=int).reshape(-1, 4)

# ---------------- step 1: verify the head / tail split against the full video
nf = len(list((full / '02').glob('t*.tif'))); nh = len(list((head / '02').glob('t*.tif'))); nt = len(list((tail / '02').glob('t*.tif')))
assert nh + nt == nf, (nh, nt, nf)
rep = dict(dataset=ds, full_frames=nf, head_frames=nh, tail_frames=nt, head_range=[0, nh - 1], tail_range=[nh, nf - 1])
full_tr = rd(full / '02_GT/TRA/man_track.txt'); head_tr = rd(head / '02_GT/TRA/man_track.txt'); tail_tr = rd(tail / '02_GT/TRA/man_track.txt')
for name, sub, off, n in (('head', head, 0, nh), ('tail', tail, nh, nt)):
    bad = 0
    for k in range(n):
        if h(sub / '02' / f't{k:03d}.tif') != h(full / '02' / f't{k + off:03d}.tif'): bad += 1
        if h(sub / '02_ST/SEG' / f'man_seg{k:03d}.tif') != h(full / '02_ST/SEG' / f'man_seg{k + off:03d}.tif'): bad += 1
        g1 = tifffile.imread(sub / '02_GT/TRA' / f'man_track{k:03d}.tif'); g0 = tifffile.imread(full / '02_GT/TRA' / f'man_track{k + off:03d}.tif')
        if not np.array_equal(g1, g0): bad += 1
    rep[f'{name}_files_mismatching_full02'] = bad
assert rep['head_files_mismatching_full02'] == 0 and rep['tail_files_mismatching_full02'] == 0, rep
# head man_track: only head frames, every row derived from the full lineage clipped to [0, nh-1]
fb = {int(L): (int(B), int(E), int(P)) for L, B, E, P in full_tr}
ok = True
for L, B, E, P in head_tr:
    B0, E0, P0 = fb[int(L)]
    ok &= (0 <= B <= E <= nh - 1) and B == max(B0, 0) and E == min(E0, nh - 1) and (P == P0 or P == 0)
exp_head_ids = {L for L, (B, E, P) in fb.items() if B <= nh - 1}
rep.update(head_track_rows=len(head_tr), head_max_E=int(head_tr[:, 2].max()), head_rows_consistent_with_full=bool(ok),
           head_ids_equal_full_ids_born_in_head=set(int(x) for x in head_tr[:, 0]) == exp_head_ids,
           tail_only_track_ids=len([L for L, (B, E, P) in fb.items() if B >= nh]))
assert ok and rep['head_max_E'] <= nh - 1 and rep['head_ids_equal_full_ids_born_in_head'], rep

OUT.mkdir(parents=True, exist_ok=True)
hd = OUT / ds; hd.mkdir(exist_ok=True)
for sub in ('02', '02_GT', '02_ST'):
    lk = hd / sub
    if not lk.exists(): lk.symlink_to(head / sub)
    assert lk.resolve() == (head / sub).resolve()
rep['head_tree'] = {sub: os.readlink(hd / sub) for sub in ('02', '02_GT', '02_ST')}

# ---------------- step 2: Delta-gap phase views of the head (logic = make_ctc_subsampled.py, src = head)
views = []
for off in range(gap):
    out = OUT / f'{ds}_d{gap}o{off}'
    for d in (out / '02', out / '02_GT/TRA', out / '02_ST/SEG'): d.mkdir(parents=True, exist_ok=True)
    frames = sorted((head / '02').glob('t*.tif'))[off::gap]
    new = {}
    for L, B, E, P in head_tr:
        B2, E2 = -(-(B - off) // gap), (E - off) // gap
        if B2 <= E2: new[int(L)] = [B2, E2, int(P)]
    for L in new: new[L][2] = new[L][2] if new[L][2] in new else 0
    (out / '02_GT/TRA/man_track.txt').write_text(''.join(f'{L} {b} {e} {p}\n' for L, (b, e, p) in sorted(new.items())))
    n_written = 0
    for k, fr in enumerate(frames):
        t = tn(fr)
        for src, dst in ((fr, out / '02' / f't{k:03d}.tif'), (head / '02_ST/SEG' / f'man_seg{t:03d}.tif', out / '02_ST/SEG' / f'man_seg{k:03d}.tif')):
            if dst.is_symlink() or dst.exists(): dst.unlink()
            dst.symlink_to(src)
        m = tifffile.imread(head / '02_GT/TRA' / f'man_track{t:03d}.tif'); m2 = np.where(np.isin(m, list(new)), m, 0).astype(np.uint16)
        dst = out / '02_GT/TRA' / f'man_track{k:03d}.tif'
        if dst.is_symlink() or dst.exists(): dst.unlink()
        if np.array_equal(m2, m) and m.dtype == np.uint16: dst.symlink_to(head / '02_GT/TRA' / f'man_track{t:03d}.tif')
        else: tifffile.imwrite(dst, m2, compression='zlib'); n_written += 1
    views.append(dict(view=str(out), offset=off, frames=len(frames), expected_frames=-(-(nh - off) // gap), source_head_frames=[tn(f) for f in frames],
                      tracks=len(new), daughters=sum(1 for v in new.values() if v[2]), gt_frames_rewritten=n_written))
    assert len(frames) == -(-(nh - off) // gap)
rep['views'] = views
json.dump(rep, open(OUT / f'{ds}_head_views.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != 'views'}))
for v in views: print(json.dumps({k: v[k] for k in ('view', 'frames', 'expected_frames', 'tracks', 'daughters', 'gt_frames_rewritten')}))
