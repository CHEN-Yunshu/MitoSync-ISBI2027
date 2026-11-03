#!/usr/bin/env python
"""kinstats.py -- MitoSync M1 shared kinetics module: the 12 validation statistics on ANY CTC directory.

One module for every M1 agent (fitter, programmer validation, baseline, paper Table 1).  CPU only, read-only on inputs.

Accepted inputs (discover()):
  (a) a CTC dataset dir with <v>/, <v>_GT/TRA/man_track.txt and <v>_ST/SEG (e.g. data/ctc/PhC-C2DL-PSC_trainhead, video 02)
  (b) a corpus CTC dir (lineage_sim / lineage_sim_v3 '<out>_ctc'): clip_*/ (images, optional), clip_*_GT/TRA, clip_*_ST/SEG;
      all clips are aggregated (statistics pooled over clips, each clip is its own window)
  (c) a multi-sequence dir such as data/ctc/Lead-BF (select sequences with --seq_regex '^(low|mid)_')
  Video 01 is refused (held-out evaluation video) unless --allow_01.

Label source: 'seg' = <v>_ST/SEG/man_seg*.tif (labels == track ids: verified for PSC/Fluo/DIC 02 and the sim twins),
'tra' = <v>_GT/TRA/man_track*.tif when these are FULL masks (Lead-BF). 'auto' picks 'tra' if the TRA labels are not
markers (median TRA area >= 0.5 x median SEG area) or SEG is missing.

Window semantics: a sequence is analysed on frames [t0, t1); the track table is clipped to the window (tracks starting
at t0 have no parent inside the window and are not entries; E is clipped to t1-1; children born >= t1 are invisible),
so a head window never sees tail information.  Frames outside the window are not read.

The 12 statistics (keys s01..s12 of summarise()['stats12']):
 s01 division rate per 1000 cell-frames at risk (births at t in (t0,t1) / sum_{t=t0}^{t1-2} n(t)); also the
     diagnostics' definition (/ all cell-frames) as per_1000_cf_diag
 s02 cycle length of complete in-window cycles (B > t0, divides in window; frames B..E); W1 vs a reference in compare();
     undefined (n=0) for windows shorter than a cycle (17-frame clips) -> validate on long programmer runs
 s03 Hertwig: mean |cos| between the division axis (daughter centroids at birth) and the mother's long axis at E-4
     (E = mother's last frame = T; 4 frames before; random = 2/pi = .637)
 s04 sister separation (centroid distance / cell diameter) at +1/+4/+8 frames after birth (median)
 s05 mitotic rounding: mother area(E)/area(E-6) and eccentricity drop ecc(E-6)-ecc(E) (median over divisions)
 s06 growth: mean d log(area)/frame over interphase steps (excl. the mother's last 6 frames and the first 3 frames of
     in-window daughters); plus complete-cycle area(E-6)/area(B+2)
 s07 speed (centroid step px/frame): median, q10, q90 + 200-point quantile vector (W1 in compare())
 s08 velocity autocorrelation C(k) = <v_t . v_t+k> / <|v|^2> at lags 1 and 4 (+ mean step-direction cosine)
 s09 nearest-neighbour centroid distance px (median/q10/q90, quantile vector for W1) + Clark-Evans ratio R
 s10 touching fraction: per-frame share of cells sharing an 8-neighbour boundary with another cell, mean over frames
 s11 entry / exit flux per 1000 cell-frames at risk (entry: parentless track with t0 < B; exit: non-mother track with
     E < t1-1), + share within 2 diameters of the border
 s12 NN linking error at gaps 1/4/8: share of cells (present at t and t+g with the same id) for which another cell at
     t+g is closer than the true continuation
Supplements: density, area fraction, MSD(1,2,4,8,16), speed by phase, mitotic intensity cue z(age), sister separation
curve k=0..12, mother rounding profiles (used by kin_fit.py).

CLI:
  $PYC kinstats.py --dataset PhC-C2DL-PSC --path $N/data/ctc/PhC-C2DL-PSC_trainhead --videos 02 --name head \
      [--frames a,b] [--window 17] [--seq_regex RE] [--max_seqs K] [--mask auto|seg|tra] [--ref ref.json] --out stats.json
  --window W splits every sequence into non-overlapping W-frame windows (clip-comparable statistics).
  --ref: a previous kinstats json (e.g. the real head or tail); adds compare() discrepancies and W1s.
Python:  import kinstats as K; S = K.run(K.discover(path, ...), dataset); D = K.compare(S, S_ref)
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import tifffile
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

N = os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace"))
# fixed per-dataset length scale (median equivalent diameter, px) used to express separations in diameters.
# PSC/Fluo: the values of lineage_sim.py / diag_20261009 (continuity with the diagnostics); DIC / Lead-BF: median
# equivalent diameter of their kin_fit source frames (theta cell_diameter_px.median 121.2 / 64.7, rounded).
# (theta PSC/Fluo medians are 13.7 / 27.1: theta's own separations use theta's diameter.)
DIAM = {'PhC-C2DL-PSC': 14.5, 'Fluo-N2DL-HeLa': 27.5, 'DIC-C2DH-HeLa': 121.0, 'Lead-BF': 65.0}
GAPS = (1, 4, 8)
LAGS = (1, 2, 4, 8)
MSD_LAGS = (1, 2, 4, 8, 16)
SIS_K = tuple(range(0, 13))
ROUND_K = tuple(range(0, 11))
CUE_AGES = tuple(range(-8, 9))
QGRID = (np.arange(200) + 0.5) / 200.0
FIELDS = ('cy', 'cx', 'area', 'ey', 'ex', 'ecc', 'imean', 'iz', 'touch', 'nnd')
CY, CX, AREA, EY, EX, ECC, IMEAN, IZ, TOUCH, NND = range(10)


def log(*a):
    print(*a, flush=True)


# ============================================================================ discovery
def _seq_spec(name, base, trk_dir, seg_dir, img_dir, frames, mask):
    return dict(name=name, trk=os.path.join(trk_dir, 'man_track.txt'), tra_dir=trk_dir, seg_dir=seg_dir,
                img_dir=img_dir, frames=frames, mask=mask)


def discover(path, videos=None, frames=None, seq_regex=None, max_seqs=None, mask='auto', allow_01=False):
    """-> list of sequence specs. frames = (a, b) half-open window applied to every sequence (None = all frames)."""
    path = path.rstrip('/')
    specs = []
    clips = sorted(c for c in glob.glob(f'{path}/clip_*') if os.path.isdir(c) and not c.endswith(('_GT', '_ST'))
                   and os.path.exists(f'{c}_GT/TRA/man_track.txt'))
    if not clips:  # clip dirs without image dirs (labels only)
        clips = sorted(c[:-3] for c in glob.glob(f'{path}/clip_*_GT') if os.path.exists(f'{c}/TRA/man_track.txt'))
    if clips:
        for c in clips:
            specs.append(_seq_spec(os.path.basename(c), path, f'{c}_GT/TRA', f'{c}_ST/SEG', c, frames, mask))
    else:
        if videos:
            vs = list(videos)
        else:
            vs = sorted(os.path.basename(d)[:-3] for d in glob.glob(f'{path}/*_GT') if os.path.exists(f'{d}/TRA/man_track.txt'))
        if seq_regex:
            vs = [v for v in vs if re.search(seq_regex, v)]
        for v in vs:
            if v == '01' and not allow_01:
                sys.exit('REFUSING: video 01 is the held-out evaluation video (pass --allow_01 for reporting only).')
            specs.append(_seq_spec(f'{os.path.basename(path)}/{v}', path, f'{path}/{v}_GT/TRA', f'{path}/{v}_ST/SEG',
                                   f'{path}/{v}', frames, mask))
    if max_seqs:
        specs = specs[:max_seqs]
    if not specs:
        raise FileNotFoundError(f'no CTC sequences under {path}')
    return specs


# ============================================================================ loading
def read_tracks(p):
    a = np.loadtxt(p, dtype=np.int64, ndmin=2)
    return {int(L): (int(B), int(E), int(P)) for L, B, E, P in a}


def touching_labels(lab):
    s = set()
    H, W = lab.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lab[max(dy, 0):H - max(-dy, 0), max(dx, 0):W - max(-dx, 0)]
        b = lab[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            s.update(np.unique(a[m]).tolist())
            s.update(np.unique(b[m]).tolist())
    return s


def _frame_files(d, prefix):
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        m = re.match(prefix + r'(\d+)\.tif$', f)
        if m:
            out[int(m.group(1))] = os.path.join(d, f)
    return out


def _choose_mask(spec, t):
    if spec['mask'] in ('seg', 'tra'):
        return spec['mask']
    seg = _frame_files(spec['seg_dir'], 'man_seg')
    tra = _frame_files(spec['tra_dir'], 'man_track')
    if t not in seg:
        return 'tra'
    if t not in tra:
        return 'seg'
    a = tifffile.imread(seg[t]); b = tifffile.imread(tra[t])
    sa = np.bincount(a.ravel())[1:]; sb = np.bincount(b.ravel())[1:]
    sa = sa[sa > 0]; sb = sb[sb > 0]
    if len(sa) == 0:
        return 'tra'
    return 'tra' if len(sb) and np.median(sb) >= 0.5 * np.median(sa) else 'seg'


def norm_image(im):
    im = im.astype(np.float32)
    if im.ndim == 3:
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    lo, hi = np.percentile(im, (1, 99.8))
    return (im - lo) / max(hi - lo, 1e-6)


def load_seq(spec):
    """Per cell-frame records of one sequence restricted to spec['frames'] (frames outside are never read)."""
    tracks = read_tracks(spec['trk'])
    seg = _frame_files(spec['seg_dir'], 'man_seg')
    tra = _frame_files(spec['tra_dir'], 'man_track')
    ts_all = sorted(set(seg) | set(tra))
    a, b = spec['frames'] if spec['frames'] else (min(ts_all), max(ts_all) + 1)
    ts = [t for t in ts_all if a <= t < b]
    msrc = _choose_mask(spec, ts[0]) if ts else 'seg'
    files = tra if msrc == 'tra' else seg
    rec = defaultdict(dict)
    frames = {}
    H = W = None
    for t in ts:
        if t not in files:
            continue
        lab = tifffile.imread(files[t]).astype(np.int32)
        H, W = lab.shape
        imp = os.path.join(spec['img_dir'], f't{t:03d}.tif')
        has_img = os.path.exists(imp)
        im = norm_image(tifffile.imread(imp)) if has_img else np.zeros(lab.shape, np.float32)
        if lab.max() == 0:
            frames[t] = dict(n=0, areafrac=0.0, touchfrac=np.nan, nnd_mean=np.nan)
            continue
        p = regionprops_table(lab, intensity_image=im, properties=('label', 'area', 'centroid', 'moments_central', 'intensity_mean'))
        labs = p['label']; cy = p['centroid-0']; cx = p['centroid-1']; area = p['area'].astype(float)
        m00 = p['moments_central-0-0']
        vr = p['moments_central-2-0'] / m00; vc = p['moments_central-0-2'] / m00; cv = p['moments_central-1-1'] / m00
        tr = vr + vc; dt = np.sqrt(np.maximum((vr - vc) ** 2 / 4 + cv ** 2, 0))
        l1 = tr / 2 + dt; l2 = tr / 2 - dt
        ey, ex = cv, l1 - vr
        bad = (np.abs(ey) + np.abs(ex)) < 1e-9
        ey = np.where(bad, np.where(vr >= vc, 1.0, 0.0), ey); ex = np.where(bad, np.where(vr >= vc, 0.0, 1.0), ex)
        nrm = np.hypot(ey, ex); ey, ex = ey / nrm, ex / nrm
        ecc = np.sqrt(np.clip(1 - l2 / np.maximum(l1, 1e-9), 0, 1))
        imean = p['intensity_mean'] if has_img else np.full(len(labs), np.nan)
        if has_img and len(labs) >= 5:
            iz = (imean - imean.mean()) / (imean.std() + 1e-6)
        else:
            iz = np.full(len(labs), np.nan)
        touch = touching_labels(lab)
        pts = np.stack([cy, cx], 1)
        if len(pts) > 1:
            dd, _ = cKDTree(pts).query(pts, k=2); nnd = dd[:, 1]
        else:
            nnd = np.full(len(pts), np.nan)
        for i, L in enumerate(labs):
            rec[int(L)][t] = (cy[i], cx[i], area[i], ey[i], ex[i], ecc[i], imean[i], iz[i], float(int(L) in touch), nnd[i])
        frames[t] = dict(n=len(labs), areafrac=float(area.sum()) / (H * W), touchfrac=len(touch) / len(labs),
                         nnd_mean=float(np.nanmean(nnd)) if len(labs) > 1 else np.nan)
    return dict(name=spec['name'], tracks=tracks, rec=dict(rec), frames=frames, H=H, W=W, mask=msrc,
                window=(a, b), has_img=os.path.isdir(spec['img_dir']))


def repair_unlinked(seq, diam, tol=1.5):
    """Annotation repair (optional, kin_fit for DIC only): a non-mother track X ending at E (not the last loaded frame)
    and exactly two parentless tracks starting at E+1 with both centroids within tol diameters of X at E -> X is their
    parent.  Returns the number of repaired divisions (tracks table modified in place)."""
    tracks, rec = seq['tracks'], seq['rec']
    last = max(seq['frames']) if seq['frames'] else -1
    starts = defaultdict(list)
    for L, (B, E, P) in tracks.items():
        if P == 0:
            starts[B].append(L)
    has_child = {P for L, (B, E, P) in tracks.items() if P > 0}
    n = 0
    for X, (B, E, P) in list(tracks.items()):
        if X in has_child or E >= last or E not in rec.get(X, {}):
            continue
        cand = [c for c in starts.get(E + 1, []) if E + 1 in rec.get(c, {})]
        x = rec[X][E]
        near = [c for c in cand if np.hypot(rec[c][E + 1][0] - x[0], rec[c][E + 1][1] - x[1]) < tol * diam]
        if len(near) == 2:
            for c in near:
                Bc, Ec, _ = tracks[c]; tracks[c] = (Bc, Ec, X)
            n += 1
    return n


# ============================================================================ accumulation
def clip_tracks(tracks, t0, t1):
    out = {}
    for L, (B, E, P) in tracks.items():
        if B >= t1 or E < t0:
            continue
        out[L] = (max(B, t0), min(E, t1 - 1), P if (P in tracks and B > t0) else 0)
    out = {L: (B, E, P if P in out else 0) for L, (B, E, P) in out.items()}
    return out


def new_acc():
    return defaultdict(list)


def accumulate(seq, t0, t1, diam, acc):
    """Raw samples of one sequence on window [t0, t1) into acc."""
    t1 = min(t1, (max(seq['frames']) + 1) if seq['frames'] else t1)
    t0 = max(t0, min(seq['frames']) if seq['frames'] else t0)
    if t1 - t0 < 2:
        return
    tracks = clip_tracks(seq['tracks'], t0, t1)
    rec, frames, H, W = seq['rec'], seq['frames'], seq['H'], seq['W']
    T = [t for t in sorted(frames) if t0 <= t < t1]
    ncell = {t: frames[t]['n'] for t in T}
    acc['n_windows'].append(1)
    acc['frames'].append(len(T))
    acc['ncells'] += [ncell[t] for t in T]
    acc['areafrac'] += [frames[t]['areafrac'] for t in T]
    acc['touchfrac'] += [frames[t]['touchfrac'] for t in T if frames[t]['n'] > 0]
    acc['cellframes'].append(sum(ncell.values()))
    acc['atrisk'].append(sum(ncell[t] for t in T if t + 1 < t1 and t + 1 in frames))
    for t in T:  # Clark-Evans R = mean NN / (0.5 / sqrt(density))
        n = ncell[t]
        if n >= 5 and np.isfinite(frames[t]['nnd_mean']):
            acc['clark_evans'].append(frames[t]['nnd_mean'] / (0.5 / np.sqrt(n / float(H * W))))
    win = lambda d: {t: r for t, r in d.items() if t0 <= t < t1}
    R = {L: win(d) for L, d in rec.items()}
    for L, d in R.items():
        for r in d.values():
            acc['area_all'].append(r[AREA])
            if np.isfinite(r[NND]):
                acc['nnd'].append(r[NND])
    children = defaultdict(list)
    for L, (B, E, P) in tracks.items():
        if P > 0:
            children[P].append(L)
    mothers = {}
    for M, ch in children.items():
        Em = tracks[M][1]
        ds = [c for c in ch if tracks[c][0] == Em + 1]
        if len(ds) >= 2:
            mothers[M] = ds
    births = defaultdict(int)
    for M in mothers:
        births[tracks[M][1] + 1] += 1
    for t in T:
        acc['frame_series'].append((len(acc['n_windows']) - 1, t, ncell[t], births.get(t, 0)))
    kd_t = {}
    for t in T:
        ids = [L for L, d in R.items() if t in d]
        if len(ids) > 1:
            kd_t[t] = (cKDTree(np.array([R[L][t][:2] for L in ids])), ids)
    # ---- per track motion
    for L, d in R.items():
        ts = sorted(d)
        if not ts or L not in tracks:
            continue
        B, E, P = tracks[L]
        is_mother = L in mothers
        born_in = P > 0 and B > t0
        steps = {}
        for t in ts:
            if t + 1 in d:
                a, b = d[t], d[t + 1]
                s = (b[0] - a[0], b[1] - a[1]); steps[t] = s
                sp = float(np.hypot(*s))
                acc['speed'].append(sp)
                phase = 'mid'
                if is_mother and t + 1 >= E - 2:
                    phase = 'premitotic'
                elif born_in and t <= B + 2:
                    phase = 'newborn'
                acc['speed_phase'].append(phase)
                # growth: interphase steps only
                if not (is_mother and t + 1 >= E - 5) and not (born_in and t <= B + 2):
                    acc['dlogA'].append(float(np.log(b[AREA] / max(a[AREA], 1.0))))
        sp_tr = [np.hypot(*steps[t]) for t in steps]
        if len(sp_tr) >= 8:
            acc['track_mean_speed'].append(float(np.mean(sp_tr)))
            acc['track_speed_var_within'].append(float(np.var(sp_tr)))
        for t, s in steps.items():
            acc['v2'].append(s[0] ** 2 + s[1] ** 2)
            for k in LAGS:
                if t + k in steps:
                    s2 = steps[t + k]
                    acc[f'vdot{k}'].append(s[0] * s2[0] + s[1] * s2[1])
                    n1, n2 = np.hypot(*s), np.hypot(*s2)
                    if n1 >= 0.5 and n2 >= 0.5:
                        acc[f'cos{k}'].append((s[0] * s2[0] + s[1] * s2[1]) / (n1 * n2))
        for k in MSD_LAGS:
            for t in ts:
                if t + k in d:
                    acc[f'msd{k}'].append((d[t + k][0] - d[t][0]) ** 2 + (d[t + k][1] - d[t][1]) ** 2)
        for g in GAPS:
            for t in ts:
                tg = t + g
                if tg in d and tg in kd_t:
                    kd, ids = kd_t[tg]
                    x = d[t][:2]
                    dtrue = float(np.hypot(d[tg][0] - x[0], d[tg][1] - x[1]))
                    kq = min(3, len(ids))
                    dd, ii = kd.query(x, k=kq)
                    alts = [float(q) for q, j in zip(np.atleast_1d(dd), np.atleast_1d(ii)) if ids[j] != L]
                    dalt = min(alts) if alts else np.inf
                    acc[f'nnerr_g{g}'].append(dalt < dtrue)
    # ---- divisions
    for M, ds in mothers.items():
        Bm, Em, Pm = tracks[M]
        tb = Em + 1
        if not (t0 < tb < t1):
            continue
        acc['n_div'].append(1)
        dm = R.get(M, {}); d1, d2 = R.get(ds[0], {}), R.get(ds[1], {})
        for k in SIS_K:
            if tb + k in d1 and tb + k in d2:
                a, b = d1[tb + k], d2[tb + k]
                acc[f'sis_px_{k}'].append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
        if tb in d1 and tb in d2:
            a, b = d1[tb], d2[tb]
            u = np.array([a[0] - b[0], a[1] - b[1]]); un = np.linalg.norm(u)
            if un > 0 and Em - 4 in dm:
                u = u / un; r = dm[Em - 4]
                c = abs(u[0] * r[EY] + u[1] * r[EX])
                acc['axis_abs_cos'].append(c)
                acc['axis_cos2psi'].append(2 * c * c - 1)
            if Em in dm:
                acc['daughter_area_over_mother_E'].append(float((a[AREA] + b[AREA]) / 2 / max(dm[Em][AREA], 1)))
            acc['daughter_asym'].append(float(min(a[AREA], b[AREA]) / max(a[AREA], b[AREA])))
        if Em in dm and Em - 6 in dm:
            acc['round_area'].append(float(dm[Em][AREA] / max(dm[Em - 6][AREA], 1)))
            acc['round_eccdrop'].append(float(dm[Em - 6][ECC] - dm[Em][ECC]))
        if Em in dm and Em - 10 in dm:
            acc['round_prof_area'].append([float(dm[Em - k][AREA] / max(dm[Em - 10][AREA], 1)) if Em - k in dm else np.nan for k in ROUND_K])
            acc['round_prof_ecc'].append([float(dm[Em - k][ECC]) if Em - k in dm else np.nan for k in ROUND_K])
            acc['round_prof_area_abs'].append([float(dm[Em - k][AREA]) if Em - k in dm else np.nan for k in ROUND_K])
        # mitotic intensity cue z(age): mother at age <0, mean of daughters at age >=0
        for age in CUE_AGES:
            t = tb + age
            if age < 0:
                if t in dm and np.isfinite(dm[t][IZ]):
                    acc[f'cue_{age}'].append(float(dm[t][IZ]))
            else:
                zz = [x[t][IZ] for x in (d1, d2) if t in x and np.isfinite(x[t][IZ])]
                if zz:
                    acc[f'cue_{age}'].append(float(np.mean(zz)))
    # ---- cycles: complete (uncensored) and censored, daughters born inside the window
    for L, (B, E, P) in tracks.items():
        if not (P > 0 and B > t0):
            continue
        d = R.get(L, {})
        if L in mothers and E + 1 < t1:
            du = E - B + 1
            acc['cycle'].append(du)
            if du >= 12 and B + 2 in d and E - 6 in d:
                acc['cycle_growth'].append(float(d[E - 6][AREA] / max(d[B + 2][AREA], 1)))
            if du >= 8:
                areas = [(t, d[t][AREA]) for t in range(B, E + 1) if t in d]
                if len(areas) >= 0.8 * du:
                    med = float(np.median([x[1] for x in areas]))
                    acc['area_age_frac'] += [((t - B) / max(E - B, 1), a_ / med) for t, a_ in areas]
        else:
            acc['cycle_censored'].append(E - B + 1)
    # ---- entries / exits
    bord = 2 * diam
    for L, (B, E, P) in tracks.items():
        d = R.get(L, {})
        if P == 0 and t0 < B < t1:
            acc['entries'].append(1)
            if B in d:
                y, x = d[B][:2]; acc['entry_border'].append(min(y, x, H - 1 - y, W - 1 - x) < bord)
        if L not in mothers and t0 <= E < t1 - 1:
            acc['exits'].append(1)
            if E in d:
                y, x = d[E][:2]; acc['exit_border'].append(min(y, x, H - 1 - y, W - 1 - x) < bord)


# ============================================================================ summary
def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def _m(v):
    return float(np.mean(v)) if len(v) else None


def qvec(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [round(float(x), 4) for x in np.quantile(v, QGRID)] if len(v) else None


def km_median(unc, cens):
    """Kaplan-Meier median of the cycle length from complete + right-censored in-window cycles (None if not reached)."""
    if not len(unc):
        return None
    ev = np.r_[np.asarray(unc, float), np.asarray(cens, float)]
    ob = np.r_[np.ones(len(unc)), np.zeros(len(cens))]
    S = 1.0
    for t in np.unique(ev[ob == 1]):
        at = (ev >= t).sum(); de = ((ev == t) & (ob == 1)).sum()
        S *= 1 - de / at
        if S <= 0.5:
            return float(t)
    return None


def summarise(acc, diam):
    cf = float(sum(acc['cellframes'])); ar = float(sum(acc['atrisk']))
    nd = len(acc['n_div'])
    S = dict(meta=dict(n_windows=len(acc['n_windows']), frames=int(sum(acc['frames'])), cell_frames=cf,
                       atrisk_cell_frames=ar, cells_per_frame_mean=_m(acc['ncells']),
                       cells_per_frame_range=[_q(acc['ncells'], 0), _q(acc['ncells'], 100)],
                       area_fraction_mean=_m(acc['areafrac']), diam_px=diam,
                       window_frames_mean=_m(acc['frames'])))
    s = {}
    s['s01_division_rate'] = dict(per_1000_cf=1000 * nd / max(ar, 1), n_div=nd,
                                  ci95=[1000 * max(nd - 1.96 * np.sqrt(nd), 0) / max(ar, 1), 1000 * (nd + 1.96 * np.sqrt(nd)) / max(ar, 1)],
                                  per_1000_cf_diag=1000 * nd / max(cf, 1))
    cyc = acc['cycle']
    s['s02_cycle_frames'] = dict(n=len(cyc), n_censored=len(acc['cycle_censored']), median=_q(cyc, 50), q25=_q(cyc, 25),
                                 q75=_q(cyc, 75), mean=_m(cyc), km_median=km_median(cyc, acc['cycle_censored']))
    ac = acc['axis_abs_cos']
    s['s03_hertwig_abs_cos'] = dict(mean=_m(ac), n=len(ac), se=float(np.std(ac) / np.sqrt(len(ac))) if len(ac) > 1 else None,
                                    mean_cos2psi=_m(acc['axis_cos2psi']))
    s['s04_sister_sep_diam'] = {f'b{k}': dict(median=(_q(acc[f'sis_px_{k}'], 50) / diam) if acc[f'sis_px_{k}'] else None,
                                              n=len(acc[f'sis_px_{k}'])) for k in (1, 4, 8)}
    s['s05_mitotic_rounding'] = dict(area_ratio_E_over_Em6=_q(acc['round_area'], 50), ecc_drop_Em6_minus_E=_q(acc['round_eccdrop'], 50),
                                     n=len(acc['round_area']))
    g = np.asarray(acc['dlogA'], float)
    s['s06_growth'] = dict(dlogA_per_frame=_m(g), se=float(g.std() / np.sqrt(len(g))) if len(g) > 1 else None, n=len(g),
                           cycle_growth_median=_q(acc['cycle_growth'], 50), n_cycles=len(acc['cycle_growth']))
    sp = np.asarray(acc['speed'], float)
    s['s07_speed_px'] = dict(median=_q(sp, 50), q10=_q(sp, 10), q90=_q(sp, 90), mean=_m(sp), n=len(sp))
    v2 = _m(acc['v2'])
    s['s08_vacf'] = dict(lag1=(_m(acc['vdot1']) / v2) if v2 and acc['vdot1'] else None,
                         lag4=(_m(acc['vdot4']) / v2) if v2 and acc['vdot4'] else None,
                         cos_lag1=_m(acc['cos1']), cos_lag4=_m(acc['cos4']))
    s['s09_nn_distance_px'] = dict(median=_q(acc['nnd'], 50), q10=_q(acc['nnd'], 10), q90=_q(acc['nnd'], 90),
                                   clark_evans_R=_m(acc['clark_evans']))
    s['s10_touching_fraction'] = _m(acc['touchfrac'])
    s['s11_flux_per_1000_cf'] = dict(entry=1000 * len(acc['entries']) / max(ar, 1), exit=1000 * len(acc['exits']) / max(ar, 1),
                                     n_entry=len(acc['entries']), n_exit=len(acc['exits']),
                                     entry_at_border=_m(acc['entry_border']), exit_at_border=_m(acc['exit_border']))
    s['s12_nn_link_error'] = {f'g{g_}': _m(acc[f'nnerr_g{g_}']) for g_ in GAPS}
    s['s12_nn_link_error'].update({f'n_g{g_}': len(acc[f'nnerr_g{g_}']) for g_ in GAPS})
    S['stats12'] = s
    S['quantiles'] = dict(speed=qvec(sp), nnd=qvec(acc['nnd']), cycle=qvec(cyc) if len(cyc) >= 5 else None)
    ph = np.array(acc['speed_phase'])
    tms = np.asarray(acc['track_mean_speed'])
    ex = dict(
        msd_px2={k: _m(acc[f'msd{k}']) for k in MSD_LAGS},
        vacf={k: ((_m(acc[f'vdot{k}']) / v2) if v2 and acc[f'vdot{k}'] else None) for k in LAGS},
        cos_turn={k: _m(acc[f'cos{k}']) for k in LAGS},
        speed_by_phase_median={k: _q(sp[ph == k], 50) for k in ('mid', 'premitotic', 'newborn') if (ph == k).any()},
        track_speed_cv=float(tms.std() / tms.mean()) if len(tms) > 5 else None,
        speed_between_track_var_frac=(float(tms.var()) / (float(tms.var()) + float(np.mean(acc['track_speed_var_within']))))
        if len(tms) > 5 else None,
        sister_sep_diam_median={k: (_q(acc[f'sis_px_{k}'], 50) / diam) if acc[f'sis_px_{k}'] else None for k in SIS_K},
        sister_sep_n={k: len(acc[f'sis_px_{k}']) for k in SIS_K},
        cue_z_by_age={a: _m(acc[f'cue_{a}']) for a in CUE_AGES},
        cue_n_by_age={a: len(acc[f'cue_{a}']) for a in CUE_AGES},
        daughter_area_over_mother_E=_q(acc['daughter_area_over_mother_E'], 50),
        daughter_asym=_q(acc['daughter_asym'], 50),
        eq_diam_px=dict(median=_q(2 * np.sqrt(np.asarray(acc['area_all']) / np.pi), 50)) if acc['area_all'] else None,
    )
    S['extra'] = ex
    return S


# ============================================================================ comparison
def w1q(qa, qb):
    if qa is None or qb is None:
        return None
    return float(np.mean(np.abs(np.asarray(qa) - np.asarray(qb))))


STAT_NAMES = {
    's01': 'division rate /1000 cf', 's02': 'cycle length W1', 's03': 'Hertwig |cos|', 's04': 'sister sep +1/+4/+8',
    's05': 'mitotic rounding', 's06': 'growth slope', 's07': 'speed W1', 's08': 'VACF lag 1/4', 's09': 'NN distance W1',
    's10': 'touching fraction', 's11': 'entry/exit flux', 's12': 'NN link error g1/4/8'}


MIN_EVENTS = 10   # s03/s04/s05: event-based statistics need >= 10 events in BOTH sets
MIN_CYCLES = 20   # s02: >= 20 complete cycles in both sets (17-frame clips and the 45/14-frame tails have none)


def compare(S, R):
    """Per-statistic discrepancy of set S from reference R (lower = closer; None if undefined in either set or if an
    event-based statistic has fewer than MIN_EVENTS / MIN_CYCLES events in either set)."""
    a, b = S['stats12'], R['stats12']
    out = {}
    ev = lambda key, nfun: min(nfun(a[key]), nfun(b[key])) >= MIN_EVENTS

    def absd(x, y):
        return None if x is None or y is None else abs(x - y)

    ra, rb = a['s01_division_rate']['per_1000_cf'], b['s01_division_rate']['per_1000_cf']
    out['s01'] = abs(np.log(max(ra, 1e-3) / max(rb, 1e-3))) if rb > 0 else None
    w = w1q(S['quantiles']['cycle'], R['quantiles']['cycle'])
    okc = min(a['s02_cycle_frames']['n'], b['s02_cycle_frames']['n']) >= MIN_CYCLES
    out['s02'] = w / b['s02_cycle_frames']['median'] if okc and w is not None and b['s02_cycle_frames']['median'] else None
    out['s03'] = absd(a['s03_hertwig_abs_cos']['mean'], b['s03_hertwig_abs_cos']['mean']) \
        if ev('s03_hertwig_abs_cos', lambda x: x['n']) else None
    dd = [absd(a['s04_sister_sep_diam'][k]['median'], b['s04_sister_sep_diam'][k]['median']) for k in ('b1', 'b4', 'b8')
          if min(a['s04_sister_sep_diam'][k]['n'], b['s04_sister_sep_diam'][k]['n']) >= MIN_EVENTS]
    dd = [x for x in dd if x is not None]
    out['s04'] = float(np.mean(dd)) if dd else None
    x1 = absd(a['s05_mitotic_rounding']['area_ratio_E_over_Em6'], b['s05_mitotic_rounding']['area_ratio_E_over_Em6'])
    x2 = absd(a['s05_mitotic_rounding']['ecc_drop_Em6_minus_E'], b['s05_mitotic_rounding']['ecc_drop_Em6_minus_E'])
    out['s05'] = None if x1 is None or x2 is None or not ev('s05_mitotic_rounding', lambda x: x['n']) else x1 + x2
    x = absd(a['s06_growth']['dlogA_per_frame'], b['s06_growth']['dlogA_per_frame'])
    out['s06'] = x
    w = w1q(S['quantiles']['speed'], R['quantiles']['speed'])
    out['s07'] = w / b['s07_speed_px']['median'] if w is not None else None
    dd = [absd(a['s08_vacf'][k], b['s08_vacf'][k]) for k in ('lag1', 'lag4')]
    out['s08'] = float(np.mean(dd)) if all(x is not None for x in dd) else None
    w = w1q(S['quantiles']['nnd'], R['quantiles']['nnd'])
    out['s09'] = w / b['s09_nn_distance_px']['median'] if w is not None else None
    out['s10'] = absd(a['s10_touching_fraction'], b['s10_touching_fraction'])
    out['s11'] = absd(a['s11_flux_per_1000_cf']['entry'], b['s11_flux_per_1000_cf']['entry']) + \
        absd(a['s11_flux_per_1000_cf']['exit'], b['s11_flux_per_1000_cf']['exit'])
    dd = [absd(a['s12_nn_link_error'][f'g{g}'], b['s12_nn_link_error'][f'g{g}']) for g in GAPS]
    out['s12'] = float(np.mean(dd)) if all(x is not None for x in dd) else None
    out['_w1_px'] = dict(speed=w1q(S['quantiles']['speed'], R['quantiles']['speed']),
                         nnd=w1q(S['quantiles']['nnd'], R['quantiles']['nnd']),
                         cycle=w1q(S['quantiles']['cycle'], R['quantiles']['cycle']))
    return out


def closer_count(D_new, D_old):
    """# statistics on which D_new (compare(new, ref)) is closer than D_old (compare(old, ref)); undefined ones skipped."""
    keys = [k for k in STAT_NAMES if D_new.get(k) is not None and D_old.get(k) is not None]
    return sum(D_new[k] < D_old[k] for k in keys), len(keys)


# ============================================================================ driver
def _load(spec):
    return load_seq(spec)


def load_all(specs, workers=16):
    if workers > 1 and len(specs) > 1:
        with Pool(min(workers, len(specs))) as p:
            return p.map(_load, specs)
    return [load_seq(s) for s in specs]


def run(specs, dataset, window=None, workers=16, loaded=None, diam=None, repair=False):
    """Load (or reuse loaded) sequences and summarise. window = W -> non-overlapping W-frame windows per sequence."""
    diam = diam or DIAM[dataset]
    seqs = loaded if loaded is not None else load_all(specs, workers)
    acc = new_acc()
    nrep = 0
    for sq in seqs:
        if repair:
            nrep += repair_unlinked(sq, diam)
        if not sq['frames']:
            continue
        a, b = min(sq['frames']), max(sq['frames']) + 1
        if window:
            for s0 in range(a, b - window + 1, window):
                accumulate(sq, s0, s0 + window, diam, acc)
        else:
            accumulate(sq, a, b, diam, acc)
    S = summarise(acc, diam)
    S['meta'].update(n_seqs=len(seqs), window=window, mask_source=sorted({s['mask'] for s in seqs}), repaired_divisions=nrep,
                     has_images=bool(all(s['has_img'] for s in seqs)))
    return S, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DIAM))
    ap.add_argument('--path', required=True)
    ap.add_argument('--videos', default=None)
    ap.add_argument('--frames', default=None, help='a,b half-open frame window')
    ap.add_argument('--window', type=int, default=None)
    ap.add_argument('--seq_regex', default=None)
    ap.add_argument('--max_seqs', type=int, default=None)
    ap.add_argument('--mask', default='auto', choices=['auto', 'seg', 'tra'])
    ap.add_argument('--name', default=None)
    ap.add_argument('--ref', default=None, help='reference kinstats json for compare()')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--allow_01', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(',')) if a.frames else None
    specs = discover(a.path, a.videos.split(',') if a.videos else None, fr, a.seq_regex, a.max_seqs, a.mask, a.allow_01)
    t0 = time.time()
    S, _ = run(specs, a.dataset, a.window, a.workers)
    S['meta'].update(name=a.name or os.path.basename(a.path.rstrip('/')), path=a.path, frames=fr, dataset=a.dataset)
    if a.ref:
        R = json.load(open(a.ref))
        S['compare_to_ref'] = dict(ref=a.ref, discrepancy=compare(S, R))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(S, open(a.out, 'w'), indent=1, default=float)
    s = S['stats12']
    log(f"{S['meta']['name']}: {len(specs)} seqs, {S['meta']['cells_per_frame_mean']:.1f} cells/frame, "
        f"div {s['s01_division_rate']['per_1000_cf']:.2f}/1000, hertwig {s['s03_hertwig_abs_cos']['mean']}, "
        f"touch {s['s10_touching_fraction']:.3f}, nn {s['s09_nn_distance_px']['median']:.1f}px, "
        f"entry {s['s11_flux_per_1000_cf']['entry']:.2f} ({time.time() - t0:.0f}s) -> {a.out}")


if __name__ == '__main__':
    main()
