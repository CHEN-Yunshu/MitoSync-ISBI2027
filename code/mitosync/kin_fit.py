#!/usr/bin/env python
"""kin_fit.py -- MitoSync M1 fitter: theta_<ds>.json (shared M1 schema) from a HEAD-ONLY CTC tree.  CPU, minutes.

  $PYC kin_fit.py --dataset PhC-C2DL-PSC --path $N/data/mitosync/kin_head/PhC-C2DL-PSC --videos 02 \
      --library $N/data/mitosync/lib_PhC-C2DL-PSC_head.pkl --out $N/eval/mitosync/theta_PhC-C2DL-PSC.json
  (Lead-BF: --path $N/data/mitosync/kin_head/Lead-BF --seq_regex '^(low|mid)_' ; DIC: --repair_unlinked;
   DIC/Lead-BF: --pooled_from theta_PhC-C2DL-PSC.json,theta_Fluo-N2DL-HeLa.json)

Every number is computed by kinstats.py (the shared statistics module) on the given frames only.  Fit rules
(also written into theta['_doc']):
  cell_diameter_px  equivalent diameter 2 sqrt(A/pi) over all cell-frames (median/q10/q90)
  division.rate_per_cell_frame   births / cell-frames at risk (kinstats s01)
  division.density_slope         Poisson GLM births(t) ~ n(t-1) exp(a + b (log n(t-1) - mean)); b reported raw, the
                                 17-frame sliding-window rate range is in _extra (use it to bound the hazard)
  division.cycle_frames          lognormal, interval-censored MLE on in-window cycles (complete: [c-.5, c+.5];
                                 still alive at window end / exit: > c-.5).  Fewer than 10 complete cycles -> sigma
                                 pooled from the PSC/Fluo fits and mu from the Euler-Lotka relation 2 E[exp(-r T)] = 1
                                 with r = fitted division rate (method recorded in _extra)
  division.hertwig_kappa         p(psi) ~ exp(kappa cos 2 psi), psi = division axis - mother long axis at T-4
                                 (T = mother's last frame), kappa moment-matched to mean |cos psi| (the validation
                                 statistic s03); the MLE kappa (I1/I0 = mean cos 2psi) is in _extra
  sisters (d0, d_inf, tau)       weighted least squares of the median sister centroid distance (in cell diameters,
                                 theta median) at k = 0..12 frames after birth: d(k) = d_inf - (d_inf - d0) exp(-k/tau)
  growth.area_vs_age_frac        median area / (median area of the same cell over its cycle) in 11 bins of age
                                 fraction (complete cycles >= 8 frames); fallback (< 5 complete cycles): exponential
                                 growth by mass balance, G = 1/(daughter area / mother area at E)
  growth.mitotic_rounding        mother profile r(k) = area(E-k)/area(E-10), k=0..10 (E = mother's last frame);
                                 base = median r(8..10); peak k* = argmax_{k<=6} |r(k)-base|; onset k0 = largest k >= k*
                                 with |r(j)-base| >= 25 % of the peak change for all j in [k*, k]; frames_before = k0+1
                                 (frames E-k0..E are mitotic); area_ratio = r(k*)/base; eccentricity_drop = ecc_base -
                                 ecc(E-k*) (negative = the mitotic object elongates, e.g. Fluo metaphase plates).
                                 Full profiles in _extra.rounding
  motion.speed_px_per_frame      centroid step quantiles
  motion.persistence_tau_frames / ou_sigma   joint fit of MSD(1,2,4,8,16) and the step autocorrelation VACF(1,2,4,8)
                                 (kinstats s08) with a fast DISCRETE-time OU + per-cell constant drift + white noise:
                                 v_t = phi v_{t-1} + s sqrt(1-phi^2) eps (per axis), x_t = x_{t-1} + v_t + u_cell,
                                 u_cell ~ N(0, d^2) per axis, observed with noise e per axis;
                                 MSD(k) = 2 s^2 [k + 2 sum_{j<k} (k-j) phi^j] + 2 d^2 k^2 + 4 e^2;
                                 persistence_tau_frames = -1/ln(phi) (0 = uncorrelated steps), ou_sigma = s (px/frame);
                                 d (drift_sigma_px_per_frame), e, phi, the drift-free fit, motility heterogeneity and
                                 pre-mitotic / newborn speed multipliers in _extra.motion
  low-n rules (n0 = 30)          n_div < n0 -> density_slope := 0 (raw in _extra); n_axis < n0 -> hertwig_kappa shrunk
                                 to the pooled PSC/Fluo kappa: (n k_hat + n0 k_pool)/(n + n0); sisters kept, flagged
  contacts                       touching fraction (frame mean) and NN centroid distance at the source density
  borders                        entries / exits per 1000 cell-frames at risk
  mitotic_cue.intensity_z_vs_age  mean within-frame z-score of the cell's mean normalised intensity (frame percentiles
                                 1/99.8, the tracker's view) vs age = t - t_birth (age < 0: mother, >= 0: daughters)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import optimize, special, stats

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinstats as K  # noqa: E402

N = K.N


def glm_poisson(X, y, off, iters=60):
    b = np.zeros(X.shape[1]); b[0] = np.log(max(y.sum(), 1) / np.exp(off).sum())
    for _ in range(iters):
        mu = np.exp(X @ b + off); g = X.T @ (y - mu); H = X.T @ (X * mu[:, None])
        b = b + np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
    mu = np.exp(X @ b + off); H = X.T @ (X * mu[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H + 1e-9 * np.eye(len(b)))))
    return b, se


def division_series(acc):
    """per window: arrays of n(t) and births(t) in frame order."""
    by = {}
    for w, t, n, b in acc['frame_series']:
        by.setdefault(w, []).append((t, n, b))
    out = []
    for w, rows in by.items():
        rows.sort()
        out.append((np.array([r[1] for r in rows], float), np.array([r[2] for r in rows], float)))
    return out


def fit_density_slope(acc):
    ys, ns = [], []
    for n, b in division_series(acc):
        if len(n) < 2:
            continue
        ns.append(n[:-1]); ys.append(b[1:])
    n = np.concatenate(ns); y = np.concatenate(ys)
    ok = n > 0
    n, y = n[ok], y[ok]
    if y.sum() < 5 or np.std(np.log(n)) < 1e-3:
        return None, None, None
    z = np.log(n) - np.log(n).mean()
    b, se = glm_poisson(np.stack([np.ones_like(z), z], 1), y, np.log(n))
    return float(b[1]), float(se[1]), float(np.exp(np.log(n).mean()))


def window_rate_range(acc, W=17):
    rates = []
    for n, b in division_series(acc):
        for s in range(0, len(n) - W + 1):
            ex = n[s:s + W - 1].sum()
            if ex > 0:
                rates.append(b[s + 1:s + W].sum() / ex)
    if not rates:
        return None
    r = np.array(rates)
    return dict(min=float(r.min()), q10=float(np.percentile(r, 10)), median=float(np.median(r)),
                q90=float(np.percentile(r, 90)), max=float(r.max()), n_windows=len(r))


def lognormal_censored_mle(unc, cens):
    unc = np.asarray(unc, float); cens = np.asarray(cens, float)

    def nll(p):
        mu, ls = p; s = np.exp(ls)
        lo = stats.norm.cdf((np.log(np.maximum(unc - 0.5, 1e-6)) - mu) / s)
        hi = stats.norm.cdf((np.log(unc + 0.5) - mu) / s)
        ll = np.sum(np.log(np.maximum(hi - lo, 1e-300)))
        if len(cens):
            ll += np.sum(stats.norm.logsf((np.log(np.maximum(cens - 0.5, 1e-6)) - mu) / s))
        return -ll
    x0 = [np.log(np.median(unc)), np.log(max(np.std(np.log(unc)), 0.05))]
    r = optimize.minimize(nll, x0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    mu, s = float(r.x[0]), float(np.exp(r.x[1]))
    # naive (complete cycles only) for reference
    return mu, s, dict(converged=bool(r.success), nll=float(r.fun), naive_mu=float(np.mean(np.log(unc))),
                       naive_sigma=float(np.std(np.log(unc))))


def euler_lotka_mu(rate, sigma):
    """mu such that 2 E[exp(-rate T)] = 1, T ~ lognormal(mu, sigma) (balanced exponential growth, no death)."""
    z, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()

    def f(mu):
        return 2 * np.sum(w * np.exp(-rate * np.exp(mu + sigma * z))) - 1
    return float(optimize.brentq(f, np.log(1.0), np.log(1e5)))


def mean_abs_cos(kappa):
    th = np.linspace(-np.pi / 2, np.pi / 2, 40001)
    w = np.exp(kappa * np.cos(2 * th))
    return float((np.abs(np.cos(th)) * w).sum() / w.sum())


def fit_kappa(mac):
    if mac is None:
        return None
    if mac <= mean_abs_cos(0.0):
        return 0.0
    if mac >= mean_abs_cos(60.0):
        return 60.0
    return float(optimize.brentq(lambda k: mean_abs_cos(k) - mac, 0.0, 60.0))


def kappa_mle(mc2):
    if mc2 is None or mc2 <= 0:
        return 0.0
    return float(optimize.brentq(lambda k: special.i1e(k) / special.i0e(k) - mc2, 1e-9, 500.0))


def fit_sisters(acc, diam):
    ks, ds, ws = [], [], []
    for k in K.SIS_K:
        v = acc[f'sis_px_{k}']
        if len(v) >= 3:
            ks.append(k); ds.append(np.median(v) / diam); ws.append(np.sqrt(len(v)))
    if len(ks) < 3:
        return None, dict(ks=ks)
    ks, ds, ws = map(np.asarray, (ks, ds, ws))
    f = lambda k, d0, dinf, tau: dinf - (dinf - d0) * np.exp(-k / tau)
    try:
        p, _ = optimize.curve_fit(f, ks, ds, p0=[ds[0], ds[-1] * 1.1, 5.0], sigma=1 / ws,
                                  bounds=([0, 0, 0.3], [10, 20, 60]), maxfev=20000)
    except Exception as e:  # noqa
        return None, dict(error=str(e))
    rmse = float(np.sqrt(np.mean((f(ks, *p) - ds) ** 2)))
    return dict(d0=float(p[0]), d_inf=float(p[1]), tau=float(p[2])), dict(
        points=[[int(k), round(float(d), 4), int(w ** 2)] for k, d, w in zip(ks, ds, ws)], rmse=rmse)


def fit_area_age(acc, slope, Tmed, dmE=None):
    pts = acc['area_age_frac']
    ncyc = len(acc['cycle'])
    if len(pts) >= 50 and ncyc >= 5:
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        b = np.clip(np.round(x * 10).astype(int), 0, 10)
        prof = [[round(i / 10, 1), round(float(np.median(y[b == i])), 4)] for i in range(11) if (b == i).sum() >= 5]
        return prof, dict(method='complete cycles', n_cycles=ncyc, n_points=len(pts))
    if dmE:  # mass balance: a newborn has dmE x the mother's pre-division area -> growth G = 1/dmE over the cycle
        G = 1.0 / dmE
        prof = [[round(i / 10, 1), round(float(G ** (i / 10 - 0.5)), 4)] for i in range(11)]
        return prof, dict(method='exponential growth G = 1/(daughter area / mother area at E) (no complete cycles)', G=G,
                          slope_implied_G=float(np.exp(slope * Tmed)) if slope is not None and Tmed else None)
    return [[0.0, 1.0], [1.0, 1.0]], dict(method='none (no data)')


def fit_rounding(acc):
    """Mother profile r(k) = median area(E-k)/area(E-10), k = 0..10 (E = mother's last frame).  base = median r(8..10);
    peak k* = argmax_{k<=6} |r(k) - base|; onset k0 = largest k >= k* with |r(j) - base| >= 25 % |r(k*) - base| for all
    j in [k*, k]; frames_before = k0 + 1 (frames E-k0..E are mitotic); area_ratio = r(k*)/base (peak change);
    eccentricity_drop = ecc_base - ecc(k*) (ecc_base = median ecc(E-8..E-10); negative = the mitotic cell elongates)."""
    P = np.array(acc['round_prof_area'], float)
    Ecc = np.array(acc['round_prof_ecc'], float)
    if len(P) < 3:
        ra, ed = K._q(acc['round_area'], 50), K._q(acc['round_eccdrop'], 50)
        return dict(frames_before=6, area_ratio=ra, eccentricity_drop=ed), dict(method='E vs E-6 (few divisions)', n=len(P))
    prof = np.nanmedian(P, 0)
    eprof = np.nanmedian(Ecc, 0)
    base = float(np.median(prof[8:11])); ebase = float(np.median(eprof[8:11]))
    kp = int(np.argmax(np.abs(prof[:7] - base)))
    full = float(prof[kp] - base)
    k0 = kp
    if abs(full) >= 0.03:
        while k0 + 1 <= 7 and (prof[k0 + 1] - base) / full >= 0.25:
            k0 += 1
    return dict(frames_before=int(k0 + 1), area_ratio=float(prof[kp] / base), eccentricity_drop=float(ebase - eprof[kp])), dict(
        method='peak of the mother area profile, onset at 25 % of the peak change', n=len(P), peak_frames_before_E=kp,
        area_profile_rel_Em10=[round(float(x), 4) for x in prof], ecc_profile=[round(float(x), 4) for x in eprof],
        k_index='k = 0..10 frames before the mother last frame E', area_ratio_E_over_Em6=K._q(acc['round_area'], 50),
        ecc_drop_Em6_minus_E=K._q(acc['round_eccdrop'], 50))


def msd_discrete(ks, phi, s2, e2, d2=0.0):
    """2-D MSD of x_t = x_{t-1} + v_t + u, v_t = phi v_{t-1} + s sqrt(1-phi^2) eps (per-axis stationary var s2), u = a
    per-cell constant drift (per-axis var d2), observed with per-axis white localisation noise var e2:
    2 s2 [k + 2 sum_{j<k} (k-j) phi^j] + 2 d2 k^2 + 4 e2."""
    out = []
    for k in ks:
        k = int(k); j = np.arange(1, k)
        out.append(2 * s2 * (k + 2 * np.sum((k - j) * phi ** j)) + 2 * d2 * k * k + 4 * e2)
    return np.array(out)


def vacf_discrete(ks, phi, s2, e2, d2=0.0):
    """step autocorrelation <s_t . s_t+k> / <|s|^2> of the same model."""
    m1 = 2 * s2 + 2 * d2 + 4 * e2
    return np.array([(2 * (s2 * phi ** k + d2) - (2 * e2 if k == 1 else 0.0)) / m1 for k in ks])


def fit_motion(msd, vacf, drift=True):
    """Joint fit of MSD(1,2,4,8,16) (relative errors) and VACF(1,2,4,8) (absolute errors, scale 0.02, weight 0.1):
    fast discrete OU (phi, s) + per-cell constant drift (d, optional) + white localisation noise (e).  Multi-start
    Nelder-Mead on (logit phi, log s2, log d2, log e2)."""
    km = np.array([k for k in K.MSD_LAGS if msd.get(k)], float); ym = np.array([msd[int(k)] for k in km])
    kv = np.array([k for k in K.LAGS if vacf.get(k) is not None], float); yv = np.array([vacf[int(k)] for k in kv])

    def unpack(q):
        return 1 / (1 + np.exp(-q[0])), np.exp(q[1]), np.exp(q[2]) if drift else 0.0, np.exp(q[3])

    def loss(q):
        phi, s2, d2, e2 = unpack(q)
        return np.sum((msd_discrete(km, phi, s2, e2, d2) / ym - 1) ** 2) + 0.1 * np.sum(((vacf_discrete(kv, phi, s2, e2, d2) - yv) / 0.02) ** 2)
    sc = np.log(ym[0] / 4)
    best = None
    for q0 in ([0, sc, sc - 3, sc - 2], [1, sc, sc - 2, sc - 1], [-1, sc + 1, sc - 4, sc - 3], [2, sc - 1, sc - 3, sc - 1],
               [0.5, sc + 0.5, sc - 1, sc - 3], [-3, sc, sc - 5, sc]):
        r = optimize.minimize(loss, q0, method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    phi, s2, d2, e2 = unpack(best.x)
    return dict(phi=float(phi), tau=float(-1 / np.log(phi)) if phi > 1e-6 else 0.0, s=float(np.sqrt(s2)),
                drift=float(np.sqrt(d2)), e=float(np.sqrt(e2)), loss=float(best.fun),
                msd_pred={int(k): round(float(v), 3) for k, v in zip(km, msd_discrete(km, phi, s2, e2, d2))},
                vacf_pred={int(k): round(float(v), 4) for k, v in zip(kv, vacf_discrete(kv, phi, s2, e2, d2))})


def fit(dataset, specs, library, repair=False, pooled=None, workers=16, n0=30):
    t0 = time.time()
    seqs = K.load_all(specs, workers)
    S, acc = K.run(specs, dataset, loaded=seqs, repair=repair)
    s = S['stats12']; ex = S['extra']
    eqd = 2 * np.sqrt(np.asarray(acc['area_all'], float) / np.pi)
    diam = float(np.median(eqd))
    rate = s['s01_division_rate']['per_1000_cf'] / 1000.0
    slope, slope_se, nref = fit_density_slope(acc)
    unc, cens = acc['cycle'], acc['cycle_censored']
    cyc_extra = dict(n_complete=len(unc), n_censored=len(cens), km_median=s['s02_cycle_frames']['km_median'],
                     complete_median=s['s02_cycle_frames']['median'])
    if len(unc) >= 10:
        mu, sg, info = lognormal_censored_mle(unc, cens)
        cyc_extra.update(method='interval-censored lognormal MLE', **info)
    else:
        pooled_sigma = (pooled or {}).get('sigma')
        sg = pooled_sigma if pooled_sigma else 0.3
        mu = euler_lotka_mu(rate, sg) if rate > 0 else float('nan')
        cyc_extra.update(method=f'Euler-Lotka from the division rate (n_complete={len(unc)} < 10), sigma pooled from '
                                f'PSC/Fluo fits' if pooled_sigma else 'Euler-Lotka, sigma default 0.3')
    cyc_extra['median_frames'] = float(np.exp(mu))
    cyc_extra['mean_frames'] = float(np.exp(mu + sg ** 2 / 2))
    cyc_extra['euler_lotka_check_mu'] = euler_lotka_mu(rate, sg) if rate > 0 else None
    mac = s['s03_hertwig_abs_cos']['mean']
    kap_raw = fit_kappa(mac)
    n_ax = s['s03_hertwig_abs_cos']['n']
    kap = kap_raw
    if n_ax < n0 and pooled and pooled.get('kappa') is not None and kap_raw is not None:
        kap = (n_ax * kap_raw + n0 * pooled['kappa']) / (n_ax + n0)  # empirical-Bayes shrinkage (low n)
    slope_raw = slope
    if s['s01_division_rate']['n_div'] < n0:
        slope = 0.0  # not identifiable from < n0 divisions
    sis, sis_extra = fit_sisters(acc, diam)
    Tmed = float(np.exp(mu)) if np.isfinite(mu) else None
    area_age, area_extra = fit_area_age(acc, s['s06_growth']['dlogA_per_frame'], Tmed, ex['daughter_area_over_mother_E'])
    rnd, rnd_extra = fit_rounding(acc)
    msd = ex['msd_px2']
    mot = fit_motion(msd, ex['vacf'], drift=True)
    mot0 = fit_motion(msd, ex['vacf'], drift=False)
    tau, sig, noise, phi = mot['tau'], mot['s'], mot['e'], mot['phi']
    cue = [[int(a), round(float(ex['cue_z_by_age'][a]), 4)] for a in K.CUE_AGES
           if ex['cue_z_by_age'][a] is not None and ex['cue_n_by_age'][a] >= 3]
    sp = s['s07_speed_px']; nn = s['s09_nn_distance_px']; fl = s['s11_flux_per_1000_cf']
    ph = ex['speed_by_phase_median']
    theta = dict(
        dataset=dataset,
        source_frames=[int(min(min(sq['frames']) for sq in seqs if sq['frames'])),
                       int(max(max(sq['frames']) for sq in seqs if sq['frames']))],
        cell_diameter_px=dict(median=diam, q10=float(np.percentile(eqd, 10)), q90=float(np.percentile(eqd, 90))),
        division=dict(rate_per_cell_frame=rate, density_slope=slope,
                      cycle_frames=dict(dist='lognormal', mu=mu, sigma=sg, n=len(unc)),
                      hertwig_kappa=kap, axis_ref='mother long axis at T-4'),
        sisters=sis if sis else dict(d0=None, d_inf=None, tau=None),
        growth=dict(area_vs_age_frac=area_age, mitotic_rounding=rnd),
        motion=dict(speed_px_per_frame=dict(median=sp['median'], q10=sp['q10'], q90=sp['q90']),
                    persistence_tau_frames=tau, ou_sigma=sig, replay_library=library),
        contacts=dict(touching_fraction=s['s10_touching_fraction'],
                      nn_distance_px=dict(median=nn['median'], q10=nn['q10'], q90=nn['q90'])),
        borders=dict(entry_per_1000_cell_frames=fl['entry'], exit_per_1000_cell_frames=fl['exit']),
        mitotic_cue=dict(intensity_z_vs_age=cue),
    )
    theta['_extra'] = dict(
        n_sequences=len(seqs), sequences=[sq['name'] for sq in seqs][:40], mask_source=S['meta']['mask_source'],
        frame_interval=1, repaired_unlinked_divisions=S['meta']['repaired_divisions'],
        cells_per_frame=dict(mean=S['meta']['cells_per_frame_mean'], range=S['meta']['cells_per_frame_range']),
        area_fraction=S['meta']['area_fraction_mean'],
        n_divisions=s['s01_division_rate']['n_div'], division_rate_ci95_per_1000=s['s01_division_rate']['ci95'],
        density_slope_raw=slope_raw, density_slope_se=slope_se, density_slope_ref_cells=nref,
        low_n_rules=f'n0={n0}: density_slope := 0 if n_div < n0; kappa shrunk to the PSC/Fluo pooled kappa with weight n0/(n+n0) if n_axis < n0; sisters kept as measured (low n flagged)',
        rate_17frame_window_per_cell_frame=window_rate_range(acc, 17),
        cycle=cyc_extra,
        hertwig=dict(mean_abs_cos=mac, n=n_ax, kappa_moment_raw=kap_raw, kappa_pooled=(pooled or {}).get('kappa'), kappa_mle_cos2psi=kappa_mle(s['s03_hertwig_abs_cos']['mean_cos2psi']),
                     implied_mean_abs_cos=mean_abs_cos(kap) if kap is not None else None,
                     convention='psi = angle(daughter centroid axis at birth, mother major axis at E-4); '
                                'p(psi) ~ exp(kappa cos 2psi) on (-pi/2, pi/2)'),
        sisters=dict(sis_extra, low_n=bool(len(acc['sis_px_4']) < n0)),
        growth=dict(area_vs_age=area_extra, dlogA_per_frame=s['s06_growth']['dlogA_per_frame'],
                    cycle_growth_ratio=s['s06_growth']['cycle_growth_median'],
                    daughter_area_over_mother_E=ex['daughter_area_over_mother_E'], daughter_asym=ex['daughter_asym']),
        rounding=rnd_extra,
        motion=dict(loc_noise_px=noise, ou_phi=phi, drift_sigma_px_per_frame=mot['drift'], fit=mot, fit_without_drift=mot0,
                    ou_adequate=bool(mot['loss'] < 0.5),
                    recommendation=('OU + drift reproduces MSD and VACF' if mot['loss'] < 0.5 else
                                    'OU + drift does NOT fit (sub-diffusive, shape-driven centroid motion): use replay_library '
                                    'trajectory segments for motion; OU numbers are only indicative'),
                    model='x_t = x_{t-1} + v_t + u_cell; v_t = phi v_{t-1} + ou_sigma sqrt(1-phi^2) N(0,1) per axis, '
                          'phi = exp(-1/persistence_tau_frames); u_cell ~ N(0, drift_sigma^2) per axis, fixed per cell; '
                          'observed centroid = x_t + N(0, loc_noise^2) per axis (omit if the masks carry their own jitter)', msd_px2=msd, vacf=ex['vacf'], speed_mean=sp['mean'],
                    speed_by_phase_median=ph,
                    premitotic_speed_mult=(ph.get('premitotic') / ph['mid']) if ph.get('premitotic') and ph.get('mid') else None,
                    newborn_speed_mult=(ph.get('newborn') / ph['mid']) if ph.get('newborn') and ph.get('mid') else None,
                    track_speed_cv=ex['track_speed_cv'], speed_between_track_var_frac=ex['speed_between_track_var_frac'],
                    motility_lognormal_s=float(np.sqrt(np.log(1 + ex['track_speed_cv'] ** 2))) if ex['track_speed_cv'] else None),
        contacts=dict(clark_evans_R=nn['clark_evans_R'], cells_per_frame_mean=S['meta']['cells_per_frame_mean']),
        borders=dict(n_entry=fl['n_entry'], n_exit=fl['n_exit'], entry_at_border=fl['entry_at_border'],
                     exit_at_border=fl['exit_at_border']),
        mitotic_cue_n_by_age=ex['cue_n_by_age'],
        kinstats_diam_px=K.DIAM[dataset],
        fit_seconds=round(time.time() - t0, 1),
    )
    theta['_doc'] = 'Fit rules' + __doc__.split('Fit rules', 1)[1]
    return theta, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(K.DIAM))
    ap.add_argument('--path', required=True)
    ap.add_argument('--videos', default=None)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--seq_regex', default=None)
    ap.add_argument('--library', required=True)
    ap.add_argument('--repair_unlinked', action='store_true')
    ap.add_argument('--pooled_from', default=None, help='comma-separated theta jsons (PSC, Fluo): cycle sigma and Hertwig '
                                                         'kappa averaged = fallback / shrinkage target for low-n datasets')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats_out', default=None)
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(',')) if a.frames else None
    specs = K.discover(a.path, a.videos.split(',') if a.videos else None, fr, a.seq_regex)
    pooled = None
    if a.pooled_from:
        th = [json.load(open(p)) for p in a.pooled_from.split(',') if os.path.exists(p)]
        if th:
            pooled = dict(sigma=float(np.mean([t['division']['cycle_frames']['sigma'] for t in th])),
                          kappa=float(np.mean([t['division']['hertwig_kappa'] for t in th])),
                          sources=[t['dataset'] for t in th])
    theta, S = fit(a.dataset, specs, a.library, a.repair_unlinked, pooled)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(theta, open(a.out, 'w'), indent=1, default=float)
    if a.stats_out:
        S['meta'].update(name=f'{a.dataset} kin_fit source', path=a.path, dataset=a.dataset)
        json.dump(S, open(a.stats_out, 'w'), indent=1, default=float)
    d = theta['division']
    print(f"[{a.dataset}] diam {theta['cell_diameter_px']['median']:.1f}px rate {1000 * d['rate_per_cell_frame']:.2f}/1000 "
          f"slope {d['density_slope']} cycle median {np.exp(d['cycle_frames']['mu']):.1f} (sigma {d['cycle_frames']['sigma']:.3f}, "
          f"n {d['cycle_frames']['n']}) kappa {d['hertwig_kappa']} sisters {theta['sisters']} "
          f"tau {theta['motion']['persistence_tau_frames']:.2f} sigma {theta['motion']['ou_sigma']:.3f} "
          f"touch {theta['contacts']['touching_fraction']:.3f} entries {theta['borders']['entry_per_1000_cell_frames']:.2f} -> {a.out}",
          flush=True)


if __name__ == '__main__':
    main()
