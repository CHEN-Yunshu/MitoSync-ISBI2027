#!/usr/bin/env python
"""MitoSync Paint control: render M1 lineages WITHOUT the generator, from 02-HEAD statistics only.

Run with the cellres env (numpy/scipy/tifffile):  PYC=CELLPOSE_PYTHON

  $PYC paint_render.py stats  --dataset PhC-C2DL-PSC            # head statistics -> $N/eval/mitosync/paint/paint_stats_<ds>.{json,npz}
  $PYC paint_render.py render --dataset PhC-C2DL-PSC --twin <M1 CTC twin root (clip_XXX_GT/TRA, clip_XXX_ST/SEG)> \
        [--clips_root <M1 clip root: meta.json gives the head reference frame + crop>] --out_root $N/data/mitosync/paint/<tag>_ctc \
        [--clips clip_000,clip_001] [--limit K] [--seed 0] [--workers 8] [--png]
  $PYC paint_render.py check  --dataset PhC-C2DL-PSC --out_root <paint root>   # realised statistics vs the head (json)

Information set: statistics come from <ds>_trainhead/02 only (PSC frames 0-254, Fluo 0-77, DIC 0-70; asserted); the
02 tail and video 01 are never read.  All statistics live in the generator's 8-bit space u = (x - lo)/(hi - lo)*255 of
the FIXED per-dataset mapping ($N/data/vace_clips/<ds>/intensity_mapping.json, the mapping gen_corpus.py inverts), so a
painted frame goes through exactly the quantisation of a generated one: u -> round(clip(u, 0, 255)) (uint8, as a
generated PNG) -> x = u8/255*(hi - lo) + lo in the source dtype (= gen_corpus.inverse_map) -> LZW t%03d.tif.

Head statistics (real head frames, ST/SEG masks; ST ids == TRA track ids in these sets):
  geometry      d_out = distance to the nearest cell pixel (+ that cell's id); k = 1 + distance to the nearest cell
                boundary pixel (boundary = a 4-neighbour with another label, cell-cell contacts included); relative
                depth rho = (k - 1)/(k_max(cell) - 1) in NB bins (0 = boundary, 1 = deepest pixel), size invariant
  background    mean map B(x): normalised Gaussian smoothing (sigma 25 px) of the pixels > 3 px from any cell, median
                over the head frames (vignetting / illumination); per-frame offset o_f = median(u - B) at d_out > R_out;
                residual r = u - B - o_f: robust std (1.4826 MAD; unsegmented debris / faint cells / protrusions are
                outliers) per frame and the masked autocorrelation C_bg(tau), |tau| <= 24 px, of r winsorised at 3 sd
                (estimator IFFT|F(r m)|^2 / IFFT|F(m)|^2 with zero padding = the spatial noise power spectrum)
  cells         contrast c_o = whole-mask mean of (u - B - o_f); per frame mean / sd of c_o; whole-mask mean M_o:
                per frame mean / sd (sd_whole: the unit of the theta cue z)
  edge / halo   P_in(rho) = sum(u - B - o_f) / sum(c_o) per relative-depth bin (interior shape / shade-off / blurred
                edge); h(d) = sum(u - B - o_f) / sum(c_nearest) at round(d_out) = d = 1..R_out (phase-contrast halo,
                fluorescence blur skirt), both pooled over the head
  texture       in-cell residual u - B - o_f - c_o P_in(rho)/<P_in>_o: std relative to rms(c_o) and its autocorrelation
  temporal      rho_bg = correlation of the background residual between frames f, f+1 (pixels background in both)
                rho_z  = lag-1 autocorrelation of a cell's contrast z-score (cells >= 9 frames from any division)
  mitotic cue   theta['mitotic_cue']['intensity_z_vs_age'] ($N/eval/mitosync/theta_<ds>.json, fitted on the head):
                mean within-frame z of the whole-mask mean intensity vs age = t - t_birth (mother < 0, daughters >= 0);
                its spread: per-age sd of the same z on the head lineage (cue_head_by_age, also the head mean for comparison)

Rendering (per clip, per frame; labels = the twin's ST/SEG, lineage = the twin's man_track.txt, so a re-timed or certified
twin can be painted as well).  Frame-level numbers are those of the clip's reference head frame (meta.json 'ref.frame', the
frame the generator is conditioned on; B is cropped with 'ref.crop'), or of a seeded random head frame without --clips_root.
  background  u = B + o_f + c_near * h(d_out) + sd_bg * (sqrt(rho_bg) N_static + sqrt(1 - rho_bg) N_t)
  cell o      u = B + o_f + c_o(t) * P_in(rho)/<P_in>_o + tex * |c_o(t)| * N'_t
              c_o(t) = c_mu + c_sd * s(age) * z_o(t) + sd_whole * g * cue(age_o(t));  z_o: AR(1) with rho_z, N(0,1);
              s(age) = sd of the head z at that age (cue_head_by_age; 1 for cells outside the -8..8 window),
              daughters continue the mother's z;  g = --cue_gain (default: cue_gain_calibrated written by `calibrate`,
              which matches the REALISED u8 cue at ages -4..0 to theta; PSC mitotic cells saturate at 255)
  N, N' are unit-variance Gaussian random fields with the measured autocorrelations (FFT filtering of white noise).
Outputs: <out_root>/<clip>/t%03d.tif (native dtype), <out_root>/<clip>_GT -> twin <clip>_GT, <clip>_ST -> twin <clip>_ST
(symlinks: the SAME labels), optional <out_root>/_png/<clip>/frame_%03d.png (u8), <out_root>/paint_manifest.json (per clip:
reference frame, seed, realised cue) and <out_root>/accepted_dirs.txt.  Restartable (clips with t<last>.tif are skipped
unless --force).
"""
import argparse, glob, hashlib, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

N = Path(os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
D = N / 'data/ctc'
HEAD_END = {'PhC-C2DL-PSC': 254, 'Fluo-N2DL-HeLa': 77, 'DIC-C2DH-HeLa': 70}   # last head frame of 02 (pre-registration 2.1)
STATS_DIR = N / 'eval/mitosync/paint'
R_OUT = {'PhC-C2DL-PSC': 12, 'Fluo-N2DL-HeLa': 12, 'DIC-C2DH-HeLa': 20}
NB = 8          # relative-depth bins
LAG = 24        # autocorrelation half window (px)
SIG_B = 25.0    # background-map smoothing (px)
CUE_W = 8       # cue ages -8..8 (theta table)
VERSION = 'paint_render v2 (2026-10-10)'
tn = lambda p: int(re.search(r'(\d+)', Path(p).stem).group(1))
log = lambda *a: print(time.strftime('%H:%M:%S'), '[paint]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ intensity mapping
def load_mapping(ds):
    p = N / 'data/vace_clips' / ds / 'intensity_mapping.json'
    m = json.load(open(p))
    return dict(lo=float(m['lo']), hi=float(m['hi']), dtype=np.dtype(m['source_dtype']), path=str(p))


def fwd(x, mp):
    """native -> generator 8-bit space (float, unclipped, so statistics are not truncated)."""
    return (x.astype(np.float64) - mp['lo']) / (mp['hi'] - mp['lo']) * 255.0


def inverse_map(u8, mp):
    """identical to gen_corpus.inverse_map: u8 (0..255) -> native dtype."""
    x = u8.astype(np.float64) / 255.0 * (mp['hi'] - mp['lo']) + mp['lo']
    info = np.iinfo(mp['dtype'])
    return np.clip(np.round(x), info.min, info.max).astype(mp['dtype'])


# ------------------------------------------------------------------------------------------------ geometry
def geometry(lab):
    """fg, d_out (distance to nearest cell pixel), lab_near (id of that cell), k (1 + distance to nearest boundary),
    rel-depth bin per fg pixel, ids, k_max per id."""
    fg = lab > 0
    if fg.any():
        d_out, (iy, ix) = ndi.distance_transform_edt(~fg, return_indices=True)
        lab_near = lab[iy, ix]
    else:
        d_out = np.full(lab.shape, 1e3); lab_near = np.zeros_like(lab)
    b = np.zeros(lab.shape, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[1:] != lab[:-1]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b &= fg
    k = ndi.distance_transform_edt(~b) + 1.0 if b.any() else np.ones(lab.shape)
    ids = np.unique(lab[fg]) if fg.any() else np.zeros(0, int)
    kmax = np.asarray(ndi.maximum(k, lab, ids)) if len(ids) else np.zeros(0)
    lut = np.ones(int(lab.max()) + 1); lut[ids] = kmax
    rel = (k - 1.0) / np.maximum(lut[lab] - 1.0, 1.0)
    rbin = np.clip((rel * NB).astype(int), 0, NB - 1)
    return fg, d_out, lab_near, k, rbin, ids


def masked_acf_sums(r, m, L=LAG):
    """numerator / denominator of the masked autocorrelation at lags |dy|,|dx| <= L (zero padded, no wrap)."""
    H, W = r.shape; P = (H + L + 1, W + L + 1)
    x = np.where(m, r, 0.0); mf = m.astype(np.float64)
    num = np.fft.irfft2(np.abs(np.fft.rfft2(x, s=P)) ** 2, s=P)
    den = np.fft.irfft2(np.abs(np.fft.rfft2(mf, s=P)) ** 2, s=P)
    iy = np.arange(-L, L + 1) % P[0]; ix = np.arange(-L, L + 1) % P[1]
    return num[np.ix_(iy, ix)], den[np.ix_(iy, ix)]


def smooth_bg(u, m, sigma=SIG_B):
    num = ndi.gaussian_filter(np.where(m, u, 0.0), sigma); den = ndi.gaussian_filter(m.astype(np.float64), sigma)
    out = num / np.maximum(den, 1e-6); out[den < 0.05] = np.nan
    return out


def _read(ds, f, mp):
    R = D / f'{ds}_trainhead' / '02'
    return fwd(tifffile.imread(R / f't{f:03d}.tif'), mp), tifffile.imread(R.parent / '02_ST/SEG' / f'man_seg{f:03d}.tif').astype(np.int32)


# ------------------------------------------------------------------------------------------------ head statistics
def _pass1(args):
    ds, f, mpd = args
    mp = dict(mpd, dtype=np.dtype(mpd['dtype']))
    u, lab = _read(ds, f, mp)
    fg = lab > 0
    d_out = ndi.distance_transform_edt(~fg) if fg.any() else np.full(lab.shape, 1e3)
    return smooth_bg(u, d_out > 3).astype(np.float32)


def _pass2(args):
    ds, f, do_acf, mpd, B = args
    mp = dict(mpd, dtype=np.dtype(mpd['dtype']))
    u, lab = _read(ds, f, mp)
    fg, d_out, lab_near, k, rbin, ids = geometry(lab)
    ro = R_OUT[ds]
    bgm = (~fg) & (d_out > ro)
    off = float(np.median(u[bgm] - B[bgm]))
    v = u - B - off                                            # background-referenced intensity
    # robust background noise: unsegmented debris / faint cells / protrusions in the background mask are outliers
    rb = v[bgm]; sd_rob = float(1.4826 * np.median(np.abs(rb - np.median(rb))))
    out = dict(frame=f, offset=off, bg_res_std=sd_rob, bg_res_std_raw=float(rb.std()), bg_med_abs=float(np.median(u[bgm])), bg_frac=float(bgm.mean()))
    vbw = np.clip(v, -3 * sd_rob, 3 * sd_rob)                  # winsorised residual for the noise spectrum
    c = np.asarray(ndi.mean(v, lab, ids)); M = np.asarray(ndi.mean(u, lab, ids))
    lutc = np.zeros(int(lab.max()) + 1); lutc[ids] = c
    # interior shape by relative depth (ratio of sums), halo by distance (normalised by the nearest cell's contrast)
    pin_s = np.zeros(NB); pin_c = np.zeros(NB)
    np.add.at(pin_s, rbin[fg], v[fg]); np.add.at(pin_c, rbin[fg], lutc[lab[fg]])
    dr = np.rint(d_out).astype(int); sel = (~fg) & (dr >= 1) & (dr <= ro)
    h_s = np.zeros(ro + 1); h_c = np.zeros(ro + 1)
    np.add.at(h_s, dr[sel], v[sel]); np.add.at(h_c, dr[sel], lutc[lab_near[sel]])
    P_f = np.where(np.abs(pin_c) > 0, pin_s / np.where(pin_c == 0, 1, pin_c), 1.0)
    s = P_f[rbin]; sm = np.asarray(ndi.mean(s, lab, ids)); lsm = np.ones(int(lab.max()) + 1); lsm[ids] = sm
    pred = lutc[lab] * s / lsm[lab]
    res_in = np.where(fg, v - pred, 0.0)
    out.update(pin_s=pin_s.tolist(), pin_c=pin_c.tolist(), h_s=h_s.tolist(), h_c=h_c.tolist(),
               c_mu=float(c.mean()), c_sd=float(c.std()), mu_whole=float(M.mean()), sd_whole=float(M.std()),
               tex_rel=float(res_in[fg].std() / max(np.sqrt(np.mean(c ** 2)), 1e-6)), n_cells=int(len(ids)),
               cell_ids=[int(i) for i in ids], cell_c=c.tolist(), cell_M=M.tolist())
    if do_acf:
        out['acf_bg'] = masked_acf_sums(np.where(bgm, vbw, 0.0), bgm)
        out['acf_in'] = masked_acf_sums(res_in, fg & (k >= 2))
        if f + 1 <= HEAD_END[ds]:
            u2, lab2 = _read(ds, f + 1, mp)
            fg2 = lab2 > 0; d2 = ndi.distance_transform_edt(~fg2); bg2 = (~fg2) & (d2 > ro)
            v2 = np.clip(u2 - B - float(np.median(u2[bg2] - B[bg2])), -3 * sd_rob, 3 * sd_rob)
            both = bgm & bg2
            out['rho_bg_pair'] = float(np.corrcoef(vbw[both], v2[both])[0, 1]) if both.sum() > 1000 else None
    return out


def cmd_stats(a):
    ds = a.dataset; mp = load_mapping(ds); R = D / f'{ds}_trainhead' / '02'
    frames = sorted(tn(p) for p in R.glob('t*.tif'))
    assert frames == list(range(HEAD_END[ds] + 1)), f'{R} must hold exactly the head frames 0..{HEAD_END[ds]}, has {frames[0]}..{frames[-1]} ({len(frames)})'
    frames = frames[::a.every]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    oj = STATS_DIR / f'paint_stats_{ds}.json'; on = STATS_DIR / f'paint_stats_{ds}.npz'
    if oj.exists() and not a.force:
        log('exists', oj); return
    import multiprocessing as mpc
    mpd = dict(mp, dtype=str(mp['dtype']))
    t0 = time.time()
    with mpc.get_context('fork').Pool(a.workers) as pool:
        Bs = pool.map(_pass1, [(ds, f, mpd) for f in frames[::a.acf_every]], chunksize=1)
    B = np.nanmedian(np.stack(Bs), 0)
    if np.isnan(B).any():   # never background in any sampled frame: fill from a coarser smoothing
        m = ~np.isnan(B); Bc = ndi.gaussian_filter(np.where(m, B, 0), 60) / np.maximum(ndi.gaussian_filter(m.astype(float), 60), 1e-6)
        B = np.where(m, B, Bc)
    with mpc.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(_pass2, [(ds, f, (i % a.acf_every == 0), mpd, B) for i, f in enumerate(frames)], chunksize=1)
    log(f'{len(res)} head frames in {time.time() - t0:.0f}s')
    ro = R_OUT[ds]
    pin_s = np.sum([r['pin_s'] for r in res], 0); pin_c = np.sum([r['pin_c'] for r in res], 0)
    h_s = np.sum([r['h_s'] for r in res], 0); h_c = np.sum([r['h_c'] for r in res], 0)
    P_in = pin_s / np.where(pin_c == 0, 1, pin_c); h = np.where(h_c != 0, h_s / np.where(h_c == 0, 1, h_c), 0.0); h[0] = 0.0
    acf = {}
    for key in ('acf_bg', 'acf_in'):
        num = np.sum([r[key][0] for r in res if key in r], 0); den = np.sum([r[key][1] for r in res if key in r], 0)
        acf[key] = num / np.maximum(den, 1e-9)
    rho_bg = [r['rho_bg_pair'] for r in res if r.get('rho_bg_pair') is not None]
    # rho_z: lag-1 autocorrelation of the contrast z of cells >= 9 frames away from any division
    tracks = np.loadtxt(R.parent / '02_GT/TRA/man_track.txt', dtype=int, ndmin=2)
    byid = {int(L): (int(B_), int(E), int(P)) for L, B_, E, P in tracks}
    nch = defaultdict(int)
    for L, (B_, E, P) in byid.items():
        if P: nch[P] += 1
    near = defaultdict(set)
    for L, (B_, E, P) in byid.items():
        if nch.get(L, 0) == 2: near[L].update(range(E - CUE_W, E + 1))
        if P and nch.get(P, 0) == 2: near[L].update(range(B_, B_ + CUE_W + 1))
    zf = {}
    for r in res:
        c = np.array(r['cell_c']); z = (c - c.mean()) / (c.std() + 1e-9); zf[r['frame']] = dict(zip(r['cell_ids'], z))
    pa, pb = [], []
    for f in zf:
        if f + 1 not in zf: continue
        for L, z in zf[f].items():
            if L in zf[f + 1] and f not in near[L] and f + 1 not in near[L]: pa.append(z); pb.append(zf[f + 1][L])
    rho_z = float(np.corrcoef(pa, pb)[0, 1]) if len(pa) > 50 else 0.9
    # spread of the cue: per-age mean / sd of the within-frame z of the WHOLE-mask mean (theta's statistic) on the head lineage
    agez = {}
    for L, (B_, E, P) in byid.items():
        if nch.get(L, 0) == 2:
            for t in range(max(B_, E - CUE_W + 1), E + 1): agez[(L, t)] = t - (E + 1)
        if P and nch.get(P, 0) == 2 and byid[P][1] == B_ - 1:
            for t in range(B_, min(E, B_ + CUE_W) + 1): agez.setdefault((L, t), t - B_)
    zz = defaultdict(list)
    for r in res:
        Mw = np.array(r['cell_M']); z = (Mw - Mw.mean()) / (Mw.std() + 1e-9)
        for L, v in zip(r['cell_ids'], z):
            a_ = agez.get((L, r['frame']))
            if a_ is not None: zz[a_].append(float(v))
    cue_head = {int(a_): dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 2 else None, n=len(v)) for a_, v in sorted(zz.items())}
    theta = json.load(open(N / 'eval/mitosync' / f'theta_{ds}.json'))
    keys = ('offset', 'bg_res_std', 'bg_res_std_raw', 'bg_med_abs', 'c_mu', 'c_sd', 'mu_whole', 'sd_whole', 'tex_rel', 'n_cells')
    per_frame = {str(r['frame']): {k: r[k] for k in keys} for r in res}
    out = dict(dataset=ds, version=VERSION, source=str(R), head_frames=[frames[0], frames[-1]], n_frames=len(frames), every=a.every,
               acf_every=a.acf_every, mapping=dict(lo=mp['lo'], hi=mp['hi'], dtype=str(mp['dtype']), path=mp['path']),
               space='u = (x - lo)/(hi - lo)*255 (generator 8-bit space, unclipped)', R_out=ro, NB=NB, lag=LAG, sigma_B=SIG_B,
               B_shape=list(B.shape), B_range=[float(np.min(B)), float(np.max(B))],
               inside_profile_rel=P_in.tolist(), halo_rel=h.tolist(), rho_bg=float(np.median(rho_bg)) if rho_bg else 0.0, rho_bg_n=len(rho_bg),
               rho_z=rho_z, rho_z_pairs=len(pa), mitotic_cue=theta['mitotic_cue']['intensity_z_vs_age'], cue_head_by_age=cue_head,
               theta=str(N / 'eval/mitosync' / f'theta_{ds}.json'),
               frame_summary={k: dict(median=float(np.median([r[k] for r in res])), q10=float(np.percentile([r[k] for r in res], 10)),
                                      q90=float(np.percentile([r[k] for r in res], 90))) for k in keys},
               per_frame=per_frame, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    np.savez_compressed(on, acf_bg=acf['acf_bg'], acf_in=acf['acf_in'], B=B.astype(np.float32), P_in=P_in, halo=h)
    json.dump(out, open(oj, 'w'), indent=1)
    log('wrote', oj, on, json.dumps({k: out[k] for k in ('rho_bg', 'rho_z', 'rho_z_pairs', 'B_range')}))
    log(json.dumps(out['frame_summary']))
    log('P_in(rel depth)', np.round(P_in, 3).tolist()); log('halo(d)/c', np.round(h, 3).tolist())


def load_stats(ds):
    oj = STATS_DIR / f'paint_stats_{ds}.json'; on = STATS_DIR / f'paint_stats_{ds}.npz'
    st = json.load(open(oj)); z = np.load(on)
    for k in ('acf_bg', 'acf_in', 'B', 'P_in', 'halo'): st[k] = z[k]
    return st


# ------------------------------------------------------------------------------------------------ rendering
def noise_filter(C, shape):
    """sqrt power spectrum of a stationary field with autocorrelation C (lags -L..L) on an FFT grid of `shape`."""
    L = (C.shape[0] - 1) // 2
    w = np.hanning(2 * L + 3)[1:-1]
    Ct = C / C[L, L] * np.outer(w, w)
    E = np.zeros(shape)
    E[np.ix_(np.arange(-L, L + 1) % shape[0], np.arange(-L, L + 1) % shape[1])] = Ct
    return np.sqrt(np.clip(np.fft.rfft2(E).real, 0, None))


def gfield(filt, shape, rng):
    f = np.fft.irfft2(np.fft.rfft2(rng.standard_normal(shape)) * filt, s=shape)
    return (f - f.mean()) / (f.std() + 1e-12)


def lineage_ages(tracks):
    """age[(L, t)] = t - t_birth for a dividing mother (< 0, last CUE_W frames) or a daughter (>= 0, first CUE_W+1 frames);
    a cell that is both (daughter that divides soon) takes its mother age."""
    byid = {int(L): (int(B), int(E), int(P)) for L, B, E, P in tracks}
    ch = defaultdict(list)
    for L, (B, E, P) in byid.items():
        if P: ch[P].append(L)
    age = {}
    for L, (B, E, P) in byid.items():
        if P and len(ch.get(P, [])) == 2 and P in byid and byid[P][1] == B - 1:
            for t in range(B, min(B + CUE_W, E) + 1): age[(L, t)] = t - B
    for P, cs in ch.items():
        if len(cs) == 2 and P in byid:
            tb = min(byid[c][0] for c in cs); B, E, _ = byid[P]
            if E == tb - 1:
                for t in range(max(E - CUE_W + 1, B), E + 1): age[(P, t)] = t - tb
    return byid, age


def clip_seed(seed, ds, clip):
    return int(hashlib.sha256(f'{seed}|{ds}|{clip}'.encode()).hexdigest()[:8], 16)


def crop_B(B, shape, crop):
    H, W = shape
    if crop:
        y0, x0 = int(crop['y0']), int(crop['x0'])
    else:
        y0, x0 = max((B.shape[0] - H) // 2, 0), max((B.shape[1] - W) // 2, 0)
    out = B[y0:y0 + H, x0:x0 + W]
    if out.shape != (H, W):   # clip larger than the real frame: edge-pad
        out = np.pad(out, ((0, H - out.shape[0]), (0, W - out.shape[1])), mode='edge')
    return out


def render_clip(args):
    ds, twin, clip, clips_root, out_root, seed, cue_gain, png, force = args
    t0 = time.time()
    st = load_stats(ds); mp = load_mapping(ds)
    gt = Path(twin) / f'{clip}_GT' / 'TRA'; seg = Path(twin) / f'{clip}_ST' / 'SEG'
    T = len(list(seg.glob('man_seg*.tif')))
    odir = Path(out_root) / clip
    if (odir / f't{T - 1:03d}.tif').exists() and not force:
        return dict(clip=clip, skipped=True)
    rng = np.random.default_rng(clip_seed(seed, ds, clip))
    ref = crop = None
    if clips_root and (Path(clips_root) / clip / 'meta.json').exists():
        r_ = json.load(open(Path(clips_root) / clip / 'meta.json')).get('ref', {}) or {}
        ref, crop = r_.get('frame'), r_.get('crop')
    head = sorted(int(k) for k in st['per_frame'])
    if ref is None or int(ref) > HEAD_END[ds]:
        ref_src = 'random_head_frame'; ref = int(rng.choice(head)); crop = None
    else:
        ref_src = 'meta.ref'; ref = int(ref)
    if str(ref) not in st['per_frame']: ref = min(head, key=lambda f: abs(f - ref))
    pf = st['per_frame'][str(ref)]
    P_in = st['P_in']; h = st['halo']; ro = st['R_out']
    cue = {int(x): float(z) for x, z in st['mitotic_cue']}
    cue_sd = {int(k): float(v['sd']) for k, v in st.get('cue_head_by_age', {}).items() if v.get('sd') is not None and v['n'] >= 20}
    rho_bg = float(np.clip(st['rho_bg'], 0, 0.99)); rho_z = float(np.clip(st['rho_z'], 0, 0.999))
    tracks = np.loadtxt(gt / 'man_track.txt', dtype=int, ndmin=2)
    byid, age = lineage_ages(tracks)
    labs = [tifffile.imread(seg / f'man_seg{t:03d}.tif').astype(np.int32) for t in range(T)]
    shape = labs[0].shape
    Bc = crop_B(st['B'], shape, crop) + pf['offset']
    fb = noise_filter(st['acf_bg'], shape); fi = noise_filter(st['acf_in'], shape)
    n_static = gfield(fb, shape, rng)
    zt = {}
    for L in sorted(byid, key=lambda L: byid[L][0]):
        B_, E, P = byid[L]
        z = rho_z * zt[(P, B_ - 1)] + np.sqrt(1 - rho_z ** 2) * rng.standard_normal() if (P and (P, B_ - 1) in zt) else rng.standard_normal()
        for t in range(B_, E + 1):
            if t > B_: z = rho_z * z + np.sqrt(1 - rho_z ** 2) * rng.standard_normal()
            zt[(L, t)] = z
    odir.mkdir(parents=True, exist_ok=True)
    pdir = Path(out_root) / '_png' / clip
    if png: pdir.mkdir(parents=True, exist_ok=True)
    hx = np.arange(len(h))
    real_z = defaultdict(list); stats = []
    for t, lab in enumerate(labs):
        fg, d_out, lab_near, k, rbin, ids = geometry(lab)
        u = Bc.copy()
        lutc = np.zeros(int(lab.max()) + 1)
        for L in ids:
            L = int(L); a = age.get((L, t))
            zb = zt.get((L, t), 0.0)
            if a is not None:   # mitotic / newborn: theta mean cue, head spread at that age (s_age), in sd_whole units
                zb *= cue_sd.get(int(a), 1.0)
            lutc[L] = pf['c_mu'] + pf['c_sd'] * zb + (pf['sd_whole'] * cue_gain * cue.get(int(a), 0.0) if a is not None else 0.0)
        bgp = ~fg
        halo = np.interp(np.clip(d_out, 0, ro), hx, h); halo[d_out > ro + 0.5] = 0.0
        nb = np.sqrt(rho_bg) * n_static + np.sqrt(1 - rho_bg) * gfield(fb, shape, rng)
        u[bgp] += lutc[lab_near[bgp]] * halo[bgp] + pf['bg_res_std'] * nb[bgp]
        if len(ids):
            s = P_in[rbin]; sm = np.asarray(ndi.mean(s, lab, ids)); lsm = np.ones(int(lab.max()) + 1); lsm[ids] = sm
            ni = gfield(fi, shape, rng)
            cc = lutc[lab[fg]]
            u[fg] += cc * s[fg] / lsm[lab[fg]] + pf['tex_rel'] * np.abs(cc) * ni[fg]
        u8 = np.clip(np.rint(u), 0, 255).astype(np.uint8)
        if len(ids) >= 5:   # realised whole-mask z of the tracker's view (kinstats definition) for the self-check
            mw = np.asarray(ndi.mean(u8.astype(float), lab, ids)); zz = (mw - mw.mean()) / (mw.std() + 1e-9)
            zl = dict(zip([int(i) for i in ids], zz))
            for L in zl:
                a = age.get((L, t))
                if a is not None: real_z[int(a)].append(float(zl[L]))
        nat = inverse_map(u8, mp)
        tifffile.imwrite(odir / f't{t:03d}.tif', nat, compression='lzw')
        if png:
            from PIL import Image
            Image.fromarray(u8).save(pdir / f'frame_{t:03d}.png')
        stats.append((float(nat.min()), float(nat.max()), float(nat.mean())))
    for sub in ('_GT', '_ST'):
        dst = Path(out_root) / f'{clip}{sub}'
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.realpath(Path(twin) / f'{clip}{sub}'), dst)
    return dict(clip=clip, frames=T, ref_frame=ref, ref_crop=crop, ref_source=ref_src, seed=clip_seed(seed, ds, clip), cue_gain=cue_gain,
                tif_dtype=str(mp['dtype']), tif_min=min(s[0] for s in stats), tif_max=max(s[1] for s in stats),
                tif_mean=float(np.mean([s[2] for s in stats])), realised_z_by_age={a: [float(np.sum(v)), len(v)] for a, v in sorted(real_z.items())},
                seconds=round(time.time() - t0, 1))


def cmd_calibrate(a):
    """cue gain g so that the REALISED whole-mask cue z (u8 frames, kinstats definition) matches theta at the peak ages.
    Saturation of bright mitotic cells at u8 = 255 (PSC) compresses the painted cue; g compensates.  Uses head-fitted
    M1 clips and the head theta only; the rendered calibration clips are deleted afterwards."""
    import shutil, multiprocessing as mpc
    ds = a.dataset; st = load_stats(ds); ages = [int(x) for x in a.ages.split(',')]
    theta = {int(x): z for x, z in st['mitotic_cue']}; target = float(np.mean([theta[x] for x in ages]))
    clips = sorted(os.path.basename(c)[:-3] for c in glob.glob(os.path.join(a.twin, 'clip_*_GT')))[:a.limit]
    base = N / 'data/mitosync/paint' / f'_calib_{ds}'
    rows = []
    for g in [float(x) for x in a.gains.split(',')]:
        od = base / f'g{g}'; od.mkdir(parents=True, exist_ok=True)
        with mpc.get_context('fork').Pool(min(a.workers, len(clips))) as pool:
            res = pool.map(render_clip, [(ds, a.twin, c, a.clips_root, str(od), 0, g, False, True) for c in clips], chunksize=1)
        agg = defaultdict(lambda: [0.0, 0])
        for r in res:
            for k_, (s_, n_) in r['realised_z_by_age'].items(): agg[int(k_)][0] += s_; agg[int(k_)][1] += n_
        real = float(np.mean([agg[x][0] / agg[x][1] for x in ages if agg[x][1]]))
        rows.append(dict(gain=g, realised_peak=real, curve={k_: v[0] / v[1] for k_, v in sorted(agg.items()) if v[1]}))
        log(f'gain {g}: realised mean z over ages {ages} = {real:.3f} (theta {target:.3f})')
        shutil.rmtree(od, ignore_errors=True)
    shutil.rmtree(base, ignore_errors=True)
    gs = np.array([r['gain'] for r in rows]); rz = np.array([r['realised_peak'] for r in rows])
    gcal = float(np.interp(target, rz, gs)) if np.all(np.diff(rz) > 0) else float(gs[np.argmin(np.abs(rz - target))])
    oj = STATS_DIR / f'paint_stats_{ds}.json'; js = json.load(open(oj))
    js.update(cue_gain_calibrated=gcal, cue_calibration=dict(ages=ages, theta_target=target, rows=rows, twin=os.path.realpath(a.twin),
                                                            n_clips=len(clips), created=time.strftime('%Y-%m-%d %H:%M:%S')))
    json.dump(js, open(oj, 'w'), indent=1)
    log(f'cue_gain_calibrated = {gcal:.3f} -> {oj}')


def cmd_render(a):
    ds = a.dataset
    if a.cue_gain is None:
        a.cue_gain = float(json.load(open(STATS_DIR / f'paint_stats_{ds}.json')).get('cue_gain_calibrated', 1.0))
        log('cue gain', a.cue_gain)
    out_root = Path(a.out_root).resolve()
    assert str(out_root).startswith(str(N / 'data/mitosync') + '/'), 'out_root must be a new path under $N/data/mitosync/'
    assert os.path.realpath(out_root) != os.path.realpath(a.twin)
    load_stats(ds)  # fail early
    clips = sorted(os.path.basename(c)[:-3] for c in glob.glob(os.path.join(a.twin, 'clip_*_GT')) if os.path.exists(os.path.join(c, 'TRA/man_track.txt')))
    if a.clips: clips = [c for c in clips if c in set(a.clips.split(','))]
    if a.limit: clips = clips[:a.limit]
    out_root.mkdir(parents=True, exist_ok=True)
    log(f'{ds}: {len(clips)} clips from {a.twin} -> {out_root}')
    jobs = [(ds, a.twin, c, a.clips_root, str(out_root), a.seed, a.cue_gain, a.png, a.force) for c in clips]
    import multiprocessing as mpc
    t0 = time.time()
    with mpc.get_context('fork').Pool(min(a.workers, max(len(jobs), 1))) as pool:
        res = pool.map(render_clip, jobs, chunksize=1)
    mf = out_root / 'paint_manifest.json'
    man = json.load(open(mf)) if mf.exists() else dict(clips={})
    for r in res:
        if not r.get('skipped'): man['clips'][r['clip']] = r
    man.update(dataset=ds, version=VERSION, twin=os.path.realpath(a.twin), clips_root=a.clips_root, seed=a.seed, cue_gain=a.cue_gain,
               stats=str(STATS_DIR / f'paint_stats_{ds}.json'), script=os.path.realpath(__file__), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    agg = defaultdict(lambda: [0.0, 0])
    for r in man['clips'].values():
        for k, (s, n) in r.get('realised_z_by_age', {}).items():
            agg[int(k)][0] += s; agg[int(k)][1] += n
    man['realised_cue_z_by_age'] = {k: [v[0] / v[1] if v[1] else None, v[1]] for k, v in sorted(agg.items())}
    man['theta_cue_z_by_age'] = {int(x): z for x, z in load_stats(ds)['mitotic_cue']}
    json.dump(man, open(mf, 'w'), indent=1)
    acc = [str(out_root / c) for c in sorted(man['clips']) if (out_root / c / 't000.tif').exists()]
    (out_root / 'accepted_dirs.txt').write_text(''.join(d + '\n' for d in acc))
    log(f'rendered {sum(1 for r in res if not r.get("skipped"))} clips in {time.time() - t0:.0f}s; {len(acc)} in accepted_dirs.txt')
    log('cue z realised (n) vs theta:', {k: (round(v[0], 2) if v[0] is not None else None, v[1], man['theta_cue_z_by_age'].get(k)) for k, v in man['realised_cue_z_by_age'].items()})


def cmd_check(a):
    """realised statistics of painted clips with the `stats` estimators, next to the head frame they were painted from."""
    ds = a.dataset; mp = load_mapping(ds); st = load_stats(ds); root = Path(a.out_root)
    man = json.load(open(root / 'paint_manifest.json'))
    rows = []
    for clip, r in sorted(man['clips'].items())[:a.limit or None]:
        Bc = crop_B(st['B'], tifffile.imread(root / clip / 't000.tif').shape, r.get('ref_crop'))
        for t in (0, r['frames'] // 2, r['frames'] - 1):
            u = fwd(tifffile.imread(root / clip / f't{t:03d}.tif'), mp)
            lab = tifffile.imread(root / f'{clip}_ST/SEG' / f'man_seg{t:03d}.tif').astype(np.int32)
            fg, d_out, lab_near, k, rbin, ids = geometry(lab); bgm = (~fg) & (d_out > st['R_out'])
            off = float(np.median(u[bgm] - Bc[bgm])); v = u - Bc - off
            c = np.asarray(ndi.mean(v, lab, ids)); M = np.asarray(ndi.mean(u, lab, ids))
            pf = st['per_frame'][str(r['ref_frame'])]
            rb = v[bgm]
            rows.append(dict(clip=clip, t=t, offset=off, bg_res_std=float(1.4826 * np.median(np.abs(rb - np.median(rb)))), c_mu=float(c.mean()), c_sd=float(c.std()),
                             mu_whole=float(M.mean()), sd_whole=float(M.std()), ref=pf))
    out = dict(dataset=ds, root=str(root), rows=rows)
    p = root / 'paint_check.json'; json.dump(out, open(p, 'w'), indent=1); log('wrote', p)
    for r in rows[:6]:
        log(r['clip'], r['t'], ' '.join(f"{k} {r[k]:.1f}/{r['ref'][k]:.1f}" for k in ('offset', 'bg_res_std', 'c_mu', 'c_sd', 'mu_whole', 'sd_whole')),
            '(painted/head ref frame)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('stats'); s.add_argument('--dataset', required=True); s.add_argument('--every', type=int, default=1)
    s.add_argument('--acf_every', type=int, default=5); s.add_argument('--workers', type=int, default=16); s.add_argument('--force', action='store_true')
    r = sp.add_parser('render'); r.add_argument('--dataset', required=True); r.add_argument('--twin', required=True)
    r.add_argument('--clips_root', default=None); r.add_argument('--out_root', required=True); r.add_argument('--clips', default=None)
    r.add_argument('--limit', type=int, default=0); r.add_argument('--seed', type=int, default=0); r.add_argument('--workers', type=int, default=8)
    r.add_argument('--cue_gain', type=float, default=None, help='default: cue_gain_calibrated from the stats json (see calibrate), else 1')
    r.add_argument('--png', action='store_true'); r.add_argument('--force', action='store_true')
    k = sp.add_parser('calibrate'); k.add_argument('--dataset', required=True); k.add_argument('--twin', required=True)
    k.add_argument('--clips_root', default=None); k.add_argument('--limit', type=int, default=12); k.add_argument('--workers', type=int, default=12)
    k.add_argument('--gains', default='1.0,1.4,1.8,2.4'); k.add_argument('--ages', default='-4,-3,-2,-1,0')
    c = sp.add_parser('check'); c.add_argument('--dataset', required=True); c.add_argument('--out_root', required=True); c.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assert a.dataset in HEAD_END, a.dataset
    {'stats': cmd_stats, 'render': cmd_render, 'check': cmd_check, 'calibrate': cmd_calibrate}[a.cmd](a)


if __name__ == '__main__':
    main()
