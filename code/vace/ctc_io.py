"""Small shared helpers for CTC-layout datasets (used by tracker_arms.py and recombine.py).

- make_slice(): write a temporal slice [start, end) of a CTC video as a new dataset folder (frames renumbered from 0,
  man_track.txt remapped, parent links that point outside the slice set to 0).
- verify_ctc(): lineage/label consistency check mimicking Trackastra's loader (np.maximum(GT marker, ST mask)):
  every track id present exactly in frames B..E, one connected component per id per frame, parents valid.
"""
import os
import re
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage as ndi

NIGHT = Path(os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
D = NIGHT / 'data/ctc'
tn = lambda p: int(re.search(r'(\d+)', Path(p).stem).group(1))


def read_tracks(path):
    t = np.loadtxt(path, dtype=int)
    return t.reshape(-1, 4)


def write_tracks(path, rows):
    Path(path).write_text(''.join(f'{int(L)} {int(B)} {int(E)} {int(P)}\n' for L, B, E, P in rows))


def make_slice(ds, video, start, end, out, overwrite=False):
    """Copy frames [start, end) of D/ds/video into out/video{,_GT/TRA,_ST/SEG}. Returns dict summary."""
    src = D / ds; out = Path(out)
    if (out / f'{video}_GT/TRA/man_track.txt').exists() and not overwrite:
        return dict(out=str(out), skipped=True)
    import shutil
    for d in (out / video, out / f'{video}_GT/TRA', out / f'{video}_ST/SEG'):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    frames = sorted((src / video).glob('t*.tif'))[start:end]
    tracks = read_tracks(src / f'{video}_GT/TRA/man_track.txt'); new = {}
    for L, B, E, P in tracks:
        B2, E2 = max(B, start) - start, min(E, end - 1) - start
        if B2 <= E2: new[L] = [B2, E2, P]
    for L in new: new[L][2] = new[L][2] if (new[L][2] in new and new[new[L][2]][1] < new[L][0]) else 0
    write_tracks(out / f'{video}_GT/TRA/man_track.txt', [(L, b, e, p) for L, (b, e, p) in sorted(new.items())])
    for k, fr in enumerate(frames):
        t = tn(fr); shutil.copy(fr, out / video / f't{k:03d}.tif')
        m = tifffile.imread(src / f'{video}_GT/TRA' / f'man_track{t:03d}.tif')
        tifffile.imwrite(out / f'{video}_GT/TRA' / f'man_track{k:03d}.tif', np.where(np.isin(m, list(new)), m, 0).astype(np.uint16))
        shutil.copy(src / f'{video}_ST/SEG' / f'man_seg{t:03d}.tif', out / f'{video}_ST/SEG' / f'man_seg{k:03d}.tif')
    return dict(out=str(out), frames=len(frames), tracks=len(new), start=start, end=end)


def verify_ctc(video_dir, max_report=5, frames=None, min_id=0):
    """min_id>0: only check connectivity/presence of ids >= min_id (e.g. only the pasted copies)."""
    """video_dir: path to <clip> (with <clip>_GT/TRA and <clip>_ST/SEG next to it). Returns dict with counts."""
    vd = Path(video_dir); gt = vd.parent / f'{vd.name}_GT/TRA'; st = vd.parent / f'{vd.name}_ST/SEG'
    tr = read_tracks(gt / 'man_track.txt'); ids_all = set(int(L) for L in tr[:, 0]); byid = {int(L): (int(B), int(E), int(P)) for L, B, E, P in tr}
    bad_parent = [L for L, (B, E, P) in byid.items() if P and (P not in byid or byid[P][1] >= B)]
    n = len(list(gt.glob('man_track*.tif'))); ts = range(n) if frames is None else frames
    bad_frames = 0; missing = 0; extra = 0; multi = 0; reports = []
    for t in ts:
        tra = tifffile.imread(gt / f'man_track{t:03d}.tif').astype(np.int32); seg = tifffile.imread(st / f'man_seg{t:03d}.tif').astype(np.int32)
        x = np.maximum(tra, seg); ids = set(int(i) for i in np.unique(x) if i); exp = set(L for L, (B, E, P) in byid.items() if B <= t <= E)
        if min_id: ids = {i for i in ids if i >= min_id}; exp = {i for i in exp if i >= min_id}
        nm = sum(1 for i in ids if ndi.label(x == i)[1] > 1)
        if ids != exp or nm:
            bad_frames += 1; missing += len(exp - ids); extra += len(ids - exp); multi += nm
            if len(reports) < max_report: reports.append(dict(t=t, missing=sorted(exp - ids)[:10], extra=sorted(ids - exp)[:10], multi_component=nm))
    from collections import Counter
    nchild = Counter(P for L, (B, E, P) in byid.items() if P)  # children per parent
    return dict(video=str(vd), frames=len(list(ts)), tracks=len(ids_all), daughters=int(sum(nchild.values())),
                divisions=int(sum(1 for v in nchild.values() if v == 2)), single_child_links=int(sum(1 for v in nchild.values() if v == 1)),
                parents_3plus=int(sum(1 for v in nchild.values() if v > 2)),
                bad_frames=bad_frames, missing_ids=missing, extra_ids=extra, multi_component=multi, bad_parents=len(bad_parent), examples=reports)


if __name__ == '__main__':
    import sys, json
    if sys.argv[1] == 'slice':
        print(json.dumps(make_slice(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], overwrite=True)))
    elif sys.argv[1] == 'verify':
        print(json.dumps(verify_ctc(sys.argv[2])))
