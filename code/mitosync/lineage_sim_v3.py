#!/usr/bin/env python
"""
lineage_sim_v3.py -- MitoSync M1: kinetics-fitted biological lineage programmer (CPU only).

Programs NEW identity-mask videos whose lineage kinetics follow theta_<ds>.json (fitted on the 02 HEAD only by
kin_fit.py) instead of replaying cut-and-paste units (lineage_sim.py).  Same output layout as lineage_sim.py, so
gen_corpus.py / relabel_by_readout.py / tracker_arms.py / kinstats.py run unchanged.

Biology (theta keys in brackets):
 (a) division hazard h = clip(rate x (n/n_ref)^density_slope, 17-frame window range of the head) x --event_budget
     [division.rate_per_cell_frame, density_slope, _extra.rate_17frame_window_per_cell_frame]; the initial cohort gets
     (age, cycle) from the stationary age structure of an exponentially growing population, f(a,T) ~ exp(-h a) g_h(T)
     1[a<T] with g_h = lognormal(mu_h, sigma) and mu_h solving Euler-Lotka 2 E[exp(-h T)] = 1 (window rate == h);
     cells born in the run draw T ~ lognormal(mu - ln event_budget, sigma) [cycle_frames] and divide at age T
 (b) Hertwig: division axis = mother long axis at T-4 (second moments of her rendered mask) + psi,
     p(psi) ~ exp(kappa cos 2 psi) [hertwig_kappa]; a real head division (mother mitotic frames + both daughters) is
     rotated by an arbitrary angle so that its daughter axis lies on the drawn axis; sisters are placed symmetrically
     about their midpoint and separate jointly, |r(k)| = rho_pair D d(k), d(k) = d_inf - (d_inf - d0) exp(-k/tau)
     [sisters; rho_pair = the real pair's mean ratio to d(k)], direction and midpoint motion from the real pair;
     independent motion after K_j = clip(3 tau, 12, 24) frames
 (c) growth: interphase area = M g(age/T) [growth.area_vs_age_frac], daughters start at
     daughter_area_over_mother_E x mother area at E (~half; real pair asymmetry kept) and reach the curve in 3 frames;
     mitotic rounding in the last frames_before frames: affine morph of the mask to the theta area / eccentricity
     profile [_extra.rounding.area_profile_rel_Em10, ecc_profile; fallback growth.mitotic_rounding]
 (d) motion: forward-only replay of real head centroid steps + shapes (arbitrary rotation, optional flip), chained
     segments, OU + drift fallback [persistence_tau_frames, ou_sigma, _extra.motion]; --interval g simulates g native
     frames per output frame (motion, divisions, lineage subsampled and remapped).  [v3.8: every active step (plan steps
     and sister-pair midpoint steps) x speed_gain, calibrated at the head density so that the rendered speed quantiles
     match the head's (contact exclusion adds displacement on top of the replayed steps: v3.7 ran ~6 % fast)]
 (e) contacts: position-based exclusion |x_i - x_j| >= c (r_i + r_j) and clustered initial placement (prob p_adj
     adjacent); (c, p_adj) calibrated on CPU so that touching fraction and NN distance match theta at the source
     density (cached, recorded in meta.json)
 (f) --overlap_mode occlude: overlaps arise from the calibrated contact physics (exclusion c, calibrated separately for
     this mode); contested pixels go to the cell on top (random z-order, daughters inherit the mother's z), and the lower
     cell loses at most overlap_max of its visible area (beyond that the contested pixels revert to the nearer centroid),
     i.e. CTC-representable occlusion.  [v3.6 also relaxed the exclusion to the overlap_max disc-lens distance: this made
     the Fluo nuclei touch 2x too often (0.21 vs 0.107), so v3.7 changes pixel ownership only; mode none is unchanged]
 (g) border entries / exits: cells live on a padded world, the canvas is a crop (CTC truncation: a cell is present
     while >= 30 % / 4 px of its mask is inside); --entries 0 renders only cells present at frame 0 and descendants
 (h) --phase_lock '1,2': every labelled birth frame t_b satisfies (t_b-1) mod 4 in the set (minimal shift of the event)

Usage:
  $PYC lineage_sim_v3.py --dataset PhC-C2DL-PSC --theta $N/eval/mitosync/theta_PhC-C2DL-PSC.json \
      --root $N/data/mitosync/kin_head --cache $N/data/mitosync/lib_PhC-C2DL-PSC_head.pkl --num_clips 20 \
      --num_frames 17 --density_scale 1.0 --event_budget 1 --interval 1 --phase_lock '1,2' --overlap_mode none \
      --overlap_max 0.15 --entries 1 --cond_version 2 --seed 0 --out $N/data/mitosync/sim_v3_PSC_x [--fig PNG]
Outputs: as lineage_sim.py (<out>/clip_%03d/{labels,cond_v2}/frame_%03d.png, lineage.txt, ref.png, meta.json;
<out>_ctc/clip_%03d_GT/TRA, clip_%03d_ST/SEG; <out>/summary.json) + <out>/clip_%03d/events.json (labelled divisions,
entries/exits, per-id kinetics).  Video 01 is never read (refused); the replay library and ref.png come from --root/--cache.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
sys.path.insert(0, f"{N}/scripts/vace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lineage_sim as LS  # noqa: E402  read-only reuse: CANVAS, MARKER_R, build_library, markers_image, contact_pairs, load_ref
import cond_render as CR  # noqa: E402
from kinstats import touching_labels  # noqa: E402  (same touching definition as the validation statistic s10)

S8 = np.ones((3, 3), bool)
VERSION = "v3.8"


def log(*a):
    print(*a, flush=True)


# ============================================================================ theta
class Theta:
    def __init__(self, path):
        th = json.load(open(path))
        self.path = path
        self.raw = th
        self.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        ex = th.get("_extra", {})
        self.dataset = th["dataset"]
        self.D = float(th["cell_diameter_px"]["median"])
        dv = th["division"]
        self.rate = float(dv["rate_per_cell_frame"])
        self.beta = float(dv.get("density_slope", 0.0) or 0.0)
        cpf = (ex.get("cells_per_frame") or {}).get("mean")
        self.cells_per_frame = float(cpf) if cpf else None
        self.n_ref = float(ex.get("density_slope_ref_cells") or cpf or 0.0) or None
        w = ex.get("rate_17frame_window_per_cell_frame") or {}
        self.rate_lo = float(w.get("min", 0.25 * self.rate))
        self.rate_hi = float(w.get("max", 4.0 * self.rate))
        cy = dv["cycle_frames"]
        self.mu, self.sigma = float(cy["mu"]), float(cy["sigma"])
        self.kappa = float(dv["hertwig_kappa"])
        s = th["sisters"]
        self.d0, self.dinf, self.tau_s = float(s["d0"]), float(s["d_inf"]), max(float(s["tau"]), 0.5)
        g = np.asarray(th["growth"]["area_vs_age_frac"], float)
        self.g_x, self.g_y = g[:, 0], g[:, 1]
        mr = th["growth"]["mitotic_rounding"]
        self.fb = max(int(mr["frames_before"]), 1)
        self.fbp = min(self.fb, 4)  # mitotic frames taken from the real division (E-4 = Hertwig reference stays interphase)
        rd = ex.get("rounding") or {}
        ap, ep = rd.get("area_profile_rel_Em10"), rd.get("ecc_profile")
        if ap and ep and len(ap) > self.fbp and np.isfinite(ap[self.fbp]) and np.isfinite(ep[self.fbp]):
            self.rf = [float(ap[k]) / float(ap[self.fbp]) for k in range(self.fbp)]  # area factor at E-k vs E-fbp
            self.ed = [float(ep[k]) - float(ep[self.fbp]) for k in range(self.fbp)]  # eccentricity change at E-k
            self.round_src = "_extra.rounding profiles"
        else:  # compact parameters, linear ramp to the peak at E
            ar, de = float(mr["area_ratio"]), float(mr["eccentricity_drop"])
            self.rf = [1 + (ar - 1) * (self.fbp - k) / self.fbp for k in range(self.fbp)]
            self.ed = [-de * (self.fbp - k) / self.fbp for k in range(self.fbp)]
            self.round_src = "growth.mitotic_rounding (linear ramp)"
        gx = ex.get("growth") or {}
        self.dau_ratio = float(gx.get("daughter_area_over_mother_E") or 0.5)
        mo = th["motion"]
        mx = ex.get("motion") or {}
        self.tau_p = float(mo["persistence_tau_frames"])
        self.ou_s = float(mo["ou_sigma"])
        self.phi = float(mx.get("ou_phi", math.exp(-1.0 / self.tau_p) if self.tau_p > 0 else 0.0))
        self.drift = float(mx.get("drift_sigma_px_per_frame", 0.0) or 0.0)
        self.ou_adequate = bool(mx.get("ou_adequate", True))
        c = th["contacts"]
        self.touch = float(c["touching_fraction"])
        nn = c["nn_distance_px"]
        self.nn_q = (float(nn["q10"]), float(nn["median"]), float(nn["q90"]))
        b = th["borders"]
        self.entry, self.exit = float(b["entry_per_1000_cell_frames"]), float(b["exit_per_1000_cell_frames"])
        self.Kj = int(min(max(round(3 * self.tau_s), 12), 24))
        self.Tmin = max(self.fb + 8, 12)

    def g(self, x):
        return float(np.interp(min(max(x, 0.0), 0.9), self.g_x, self.g_y))

    def dsis(self, k):
        return self.dinf - (self.dinf - self.d0) * math.exp(-k / self.tau_s)

    def msd_pred(self, k):
        s, phi, d = self.ou_s, self.phi, self.drift
        acc = k + 2 * sum((k - j) * phi ** j for j in range(1, k))
        return 2 * s * s * acc + 2 * d * d * k * k


def euler_lotka_mu(h, sigma):
    """mu such that 2 E[exp(-h T)] = 1 for T ~ lognormal(mu, sigma) (Gauss-Hermite)."""
    z, w = np.polynomial.hermite_e.hermegauss(64)
    w = w / w.sum()
    f = lambda mu: 2 * float(np.sum(w * np.exp(-h * np.exp(mu + sigma * z)))) - 1
    lo, hi = -2.0, 12.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if f(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# ============================================================================ geometry
def rotv(vy, vx, phi, flip=False):
    """rotate a (y, x) vector by phi in the (x + i y) plane after an optional y-flip (same convention as crop_affine)."""
    if flip:
        vy = -vy
    c, s = math.cos(phi), math.sin(phi)
    return s * vx + c * vy, c * vx - s * vy


def crop_affine(phi, flip, scale, Smorph=None):
    """2x2 (x, y) matrix: scale * R(phi) * F(flip) * Smorph."""
    c, s = math.cos(phi), math.sin(phi)
    A = np.array([[c, -s], [s, c]], np.float64)
    if flip:
        A = A @ np.array([[1.0, 0.0], [0.0, -1.0]])
    if Smorph is not None:
        A = A @ Smorph
    return scale * A


def moments(crop):
    ys, xs = np.nonzero(crop)
    cy, cx = float(ys.mean()), float(xs.mean())
    vr = float(((ys - cy) ** 2).mean())
    vc = float(((xs - cx) ** 2).mean())
    cv = float(((ys - cy) * (xs - cx)).mean())
    tr = vr + vc
    dt = math.sqrt(max((vr - vc) ** 2 / 4 + cv ** 2, 0.0))
    l1, l2 = tr / 2 + dt, tr / 2 - dt
    ey, ex = cv, l1 - vr
    if abs(ey) + abs(ex) < 1e-9:
        ey, ex = (1.0, 0.0) if vr >= vc else (0.0, 1.0)
    nr = math.hypot(ey, ex)
    ecc = math.sqrt(min(max(1 - l2 / max(l1, 1e-9), 0.0), 1.0))
    return dict(cy=cy, cx=cx, area=int(len(ys)), ey=ey / nr, ex=ex / nr, ecc=ecc)


def largest_cc(m):
    cc, k = ndi.label(m)
    if k > 1:
        sizes = ndi.sum(m, cc, range(1, k + 1))
        m = cc == (1 + int(np.argmax(sizes)))
    return m


def warp_crop(crop, A, fy=0.0, fx=0.0):
    """Affine-transform a bool crop about its centroid with sub-pixel placement: the source centroid lands at
    (anchor + fy, anchor + fx) of the returned crop.  Returns (bool crop trimmed, ay, ax): paste the crop with its
    top-left at (floor(y) - ay, floor(x) - ax) so that the source centroid sits at the exact position (y, x) (removes
    the +-0.5 px paste quantisation of lineage_sim.py, which would add white noise to every centroid step)."""
    h, w = crop.shape
    ys, xs = np.nonzero(crop)
    cy, cx = ys.mean(), xs.mean()
    corners = np.array([[-0.5 - cx, -0.5 - cy], [w - 0.5 - cx, -0.5 - cy], [-0.5 - cx, h - 0.5 - cy],
                        [w - 0.5 - cx, h - 0.5 - cy]])
    tc = corners @ A.T
    mn = np.floor(tc.min(0)) - 2
    mx = np.ceil(tc.max(0)) + 3
    Wo, Ho = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = A
    M[:, 2] = -A @ np.array([cx, cy]) - mn + np.array([fx, fy])
    out = cv2.warpAffine(crop.astype(np.float32), M, (Wo, Ho), flags=cv2.INTER_LINEAR, borderValue=0) >= 0.5
    if out.sum() < 4:  # tiny masks: nearest neighbour keeps them alive
        out = cv2.warpAffine(crop.astype(np.uint8), M, (Wo, Ho), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        if not out.any():
            out[int(-mn[1]), int(-mn[0])] = True
    out = largest_cc(out)
    yy, xx = np.nonzero(out)
    y0, x0 = yy.min(), xx.min()
    out = np.ascontiguousarray(out[y0:yy.max() + 1, x0:xx.max() + 1])
    return out, int(-mn[1]) - int(y0), int(-mn[0]) - int(x0)


def lens_distance(ri, rj, F):
    """centre distance at which two discs overlap by F x the smaller disc area (vectorised bisection)."""
    ri, rj = np.asarray(ri, float), np.asarray(rj, float)
    target = F * np.pi * np.minimum(ri, rj) ** 2
    lo, hi = np.abs(ri - rj), ri + rj

    def lens(d, a, b):
        d = np.maximum(d, 1e-6)
        c1 = np.clip((d * d + a * a - b * b) / (2 * d * a), -1, 1)
        c2 = np.clip((d * d + b * b - a * a) / (2 * d * b), -1, 1)
        k = np.maximum((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b), 0)
        return a * a * np.arccos(c1) + b * b * np.arccos(c2) - 0.5 * np.sqrt(k)

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        big = lens(mid, ri, rj) > target  # too much overlap -> move apart
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    return 0.5 * (lo + hi)


# ============================================================================ library pool
def contiguous_fwd(fr, start):
    out = []
    t = start
    while t in fr:
        out.append(t)
        t += 1
    return out


def contiguous_bwd(fr, end):
    out = []
    t = end
    while t in fr:
        out.append(t)
        t -= 1
    return out[::-1]


class LibPool:
    """Head replay library -> interphase pieces (forward only; real mitotic and newborn frames excluded) and real
    division templates (mother pre-mitotic run + mitotic frames + both daughters' runs from birth)."""

    def __init__(self, lib, th):
        T = lib["tracks"]
        self.T = T
        fbp = th.fbp
        mothers = {P: ch for P, ch in lib["mothers"].items() if P in T}
        self.is_mother = set(mothers)
        self._mom = {}
        self.divs = []
        for M, ch in mothers.items():
            tm = T[M]
            E, fr = tm["E"], tm["fr"]
            if not all((E - k) in fr for k in range(fbp)):
                continue
            ds = [d for d in ch if T[d]["B"] == E + 1 and (E + 1) in T[d]["fr"]]
            if len(ds) < 2:
                continue
            d1, d2 = ds[0], ds[1]
            f1, f2 = contiguous_fwd(T[d1]["fr"], E + 1), contiguous_fwd(T[d2]["fr"], E + 1)
            if len(f1) < 3 or len(f2) < 3:
                continue
            c1, c2 = T[d1]["fr"][E + 1], T[d2]["fr"][E + 1]
            uy, ux = c1[0] - c2[0], c1[1] - c2[1]
            if math.hypot(uy, ux) < 0.5:
                continue
            mid = (0.5 * (c1[0] + c2[0]), 0.5 * (c1[1] + c2[1]))
            off = (mid[0] - fr[E][0], mid[1] - fr[E][1])
            pre = contiguous_bwd(fr, E - fbp)
            # pair separation residual relative to theta's d(k) (real pair variability)
            rr = []
            for k in range(0, 13):
                t = E + 1 + k
                if t <= f1[-1] and t <= f2[-1]:
                    a, b = T[d1]["fr"][t], T[d2]["fr"][t]
                    rr.append(math.hypot(a[0] - b[0], a[1] - b[1]) / (th.D * th.dsis(k)))
            rho = float(np.clip(np.mean(rr), 0.4, 2.5)) if rr else 1.0
            self.divs.append(dict(M=M, E=E, d=(d1, d2), f=(f1, f2), off=off, pre=pre, u0=(uy, ux), rho=rho))
        pieces = []
        for L, frames in lib["runs"]:
            tk = T[L]
            keep = [f for f in frames if not (L in self.is_mother and f > tk["E"] - fbp)
                    and not (tk["P"] > 0 and f <= tk["B"] + 2)]
            cur = []
            for f in keep:
                if cur and f != cur[-1] + 1:
                    if len(cur) >= 3:
                        pieces.append((L, cur))
                    cur = []
                cur.append(f)
            if len(cur) >= 3:
                pieces.append((L, cur))
        self.pieces = pieces
        self.plen = np.array([len(p[1]) for p in pieces])
        Ms = []
        for L, tk in T.items():
            if len(tk["fr"]) >= 8:
                Ms.append(float(np.median([c.sum() for _, _, c in tk["fr"].values()])))
        self.M_pool = np.array(Ms)
        self.rho = np.array([d["rho"] for d in self.divs])

    def mom(self, L, f):
        k = (L, f)
        m = self._mom.get(k)
        if m is None:
            m = moments(self.T[L]["fr"][f][2])
            self._mom[k] = m
        return m

    def area(self, L, f):
        return self.mom(L, f)["area"]

    def sample_any(self, rng, n):
        """forward interphase frames for n frames -> list of (L, frames)."""
        out = []
        while n > 0:
            want = min(n, 24)
            idx = np.nonzero(self.plen >= want)[0]
            if not len(idx):
                idx = np.arange(len(self.pieces))
            L, fr = self.pieces[int(rng.choice(idx))]
            take = min(n, len(fr))
            s0 = int(rng.integers(0, len(fr) - take + 1))
            out.append((L, fr[s0:s0 + take]))
            n -= take
        return out


class Seg:
    """a run of real library frames assigned to consecutive simulated frames, with its rotation / flip and the log-area
    trend of the source frames (removed when the interphase growth curve is imposed)."""
    __slots__ = ("kind", "L", "frames", "rot", "flip", "alpha", "beta", "exact_upto")

    def __init__(self, kind, L, frames, rot, flip, pool, exact_upto=-1, fit_from=0):
        self.kind, self.L, self.frames, self.rot, self.flip = kind, L, list(frames), float(rot), bool(flip)
        self.exact_upto = exact_upto  # segment indices <= exact_upto use exact area targets
        fr = self.frames[fit_from:]
        if len(fr) >= 4:
            la = np.log([max(pool.area(L, f), 1) for f in fr])
            b, a = np.polyfit(np.asarray(fr, float), la, 1)
            self.alpha, self.beta = float(a), float(b)
        elif fr:
            self.alpha, self.beta = float(np.mean(np.log([max(pool.area(L, f), 1) for f in fr]))), 0.0
        else:
            self.alpha, self.beta = 0.0, 0.0


# ============================================================================ cells
class Cell:
    __slots__ = ("wid", "B", "T", "nb", "parent", "start", "end", "M", "A_birth", "phi", "flip", "z", "plan", "pos",
                 "v", "u", "div_tp", "phi_d", "psi", "axis", "birth_tp", "birth_slot", "phi_b", "ecc_ref", "children",
                 "init_young")

    def __init__(self, wid):
        self.wid = wid
        self.parent = 0
        self.nb = None
        self.A_birth = None
        self.plan = {}
        self.pos = {}
        self.children = []
        self.div_tp = None
        self.birth_tp = None
        self.birth_slot = 0
        self.phi_d = None
        self.psi = None
        self.axis = None
        self.ecc_ref = None
        self.init_young = False


class Sim:
    """One clip: world simulation on native frames + rendering of the output frames."""

    def __init__(self, pool, th, args, seed, calib):
        self.pool, self.th, self.args = pool, th, args
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.c_ex, self.p_adj = float(calib["c"]), float(calib["p_adj"])
        self.h_gain = float(calib.get("hazard_gain", 1.0))
        self.v_gain = float(calib.get("speed_gain", 1.0))  # v3.8: calibrated motion gain (1 = v3.7)
        self.H, self.W = LS.CANVAS[args.dataset]
        self.I = int(args.interval)
        self.nf_out = int(args.num_frames)
        self.NF = (self.nf_out - 1) * self.I + 1
        self.occ = args.overlap_mode == "occlude"
        self.F = float(args.overlap_max) if self.occ else 0.0
        self.allowed = set(int(x) for x in args.phase_lock.split(",") if x.strip() != "") if args.phase_lock else set()
        self.cells = {}
        self.next_wid = 1
        self.crop_cache = {}
        self.events_native = []  # (mother wid, d1 wid, d2 wid, nb, psi, phase shift)
        self.lock_shifts = []
        self.pairs = []

    # ------------------------------------------------------------------ kinetics helpers
    def hazard(self, n_canvas):
        th, a = self.th, self.args
        nat_area = float(self.pool_lib_HW[0] * self.pool_lib_HW[1])
        n_equiv = n_canvas * nat_area / float(self.H * self.W)
        # anchored so that density_scale 1 (n = the head's mean cells/frame) gives theta's pooled rate; the GLM slope
        # modulates the hazard around it (bounded by the head's 17-frame window range)
        n_anchor = th.cells_per_frame or th.n_ref
        f = (n_equiv / n_anchor) ** th.beta if (n_anchor and th.beta) else 1.0
        h0 = min(max(th.rate * f, th.rate_lo), th.rate_hi)
        # hazard_gain (calibrated): compensates labelled divisions lost at the canvas border (one daughter outside),
        # so that the labelled rate per at-risk cell-frame (kinstats s01) equals theta's rate at density 1
        return h0, h0 * self.h_gain * float(a.event_budget), n_equiv

    def lock(self, nb, lo):
        """phase-lock a native birth frame: output frame t_b = ceil(nb / I) must satisfy (t_b-1) mod 4 in allowed."""
        if not self.allowed:
            return nb
        I, nfo = self.I, self.nf_out

        def tb(x):
            return -(-x // I)

        def ok_in(x):
            j = tb(x)
            return 1 <= j <= nfo - 1 and ((j - 1) % 4) in self.allowed

        j0 = tb(nb)
        if j0 >= nfo or j0 < 1:
            return nb  # outside the clip: not a labelled birth
        if ok_in(nb):
            return nb
        sgn = [1, -1] if self.rng.random() < 0.5 else [-1, 1]
        for s in range(1, 4 * I + 1):
            for sg in sgn:
                x = nb + sg * s
                if x >= lo and ok_in(x):
                    self.lock_shifts.append(sg * s)
                    return x
        # no allowed in-clip position >= lo (fast cycler near the clip end): move the event past the last frame
        # (it is then not a labelled birth of this clip)
        x = max(nb, (nfo - 1) * I + 1)
        self.lock_shifts.append(x - nb)
        return x

    def sample_T_newborn(self):
        th = self.th
        mu = th.mu - math.log(max(float(self.args.event_budget), 1e-3))
        T = int(round(math.exp(mu + th.sigma * self.rng.standard_normal())))
        return max(T, th.Tmin)

    # ------------------------------------------------------------------ area targets
    def area_interphase(self, c, t):
        a = t - c.B
        Ag = c.M * self.th.g(a / float(c.T))
        if c.A_birth is not None and a < 3:
            return c.A_birth + (Ag - c.A_birth) * a / 3.0
        return Ag

    def area_target(self, c, t):
        fbp = self.th.fbp
        if c.nb is not None:
            E = c.nb - 1
            if E - fbp < t <= E:
                return self.area_interphase(c, E - fbp) * self.th.rf[E - t]
        return self.area_interphase(c, t)

    # ------------------------------------------------------------------ plans
    def build_plan(self, c):
        """frames p0..c.end -> (Seg, index).  Order: newborn piece (real daughter run of the birth template), filler
        interphase pieces, pre-mitotic run of the own division template, mitotic frames (rotated to the Hertwig axis)."""
        pool, th, rng = self.pool, self.th, self.rng
        fbp = th.fbp
        dividing = c.nb is not None and c.nb <= self.NF - 1
        E = c.nb - 1 if dividing else None
        p0 = min(c.start, E - 6) if dividing else c.start
        end = c.end
        plan = {}
        lim_new = (E - fbp) if dividing else end  # newborn piece must end before the mitotic frames
        t = p0
        if c.birth_tp is not None:
            tp = c.birth_tp
            L = tp["d"][c.birth_slot]
            fr = tp["f"][c.birth_slot]
            k0 = p0 - c.B  # age at the plan start
            avail = fr[k0:k0 + th.Kj + 8] if k0 < len(fr) else []
            n_new = max(0, min(len(avail), lim_new - p0 + 1 - (4 if dividing else 0)))
            if n_new > 0:
                ex_upto = max(-1, 2 - k0)  # ages 0..2 use exact (blended) targets
                sg = Seg("new", L, avail[:n_new], c.phi_b, False, pool, exact_upto=ex_upto, fit_from=max(0, 3 - k0))
                for i in range(n_new):
                    plan[t + i] = (sg, i)
                t += n_new
        tail = []
        if dividing:
            tp = c.div_tp
            pre = tp["pre"]
            n_pre = min(len(pre), (E - fbp) - t + 1)
            t_pre0 = E - fbp - n_pre + 1
        else:
            n_pre, t_pre0 = 0, end + 1
        # filler between t and t_pre0-1
        n_fill = t_pre0 - t
        if n_fill > 0:
            for L, frs in pool.sample_any(rng, n_fill):
                sg = Seg("any", L, frs, c.phi, c.flip, pool)
                for i in range(len(frs)):
                    plan[t + i] = (sg, i)
                t += len(frs)
        if dividing:
            if n_pre > 0:
                sg = Seg("pre", tp["M"], pre[len(pre) - n_pre:], c.phi, c.flip, pool)
                for i in range(n_pre):
                    plan[t + i] = (sg, i)
                t += n_pre
            # Hertwig: mother long axis at E-4 (+ psi ~ exp(kappa cos 2psi))
            sgr, ir = plan[E - 4]
            m = pool.mom(sgr.L, sgr.frames[ir])
            ay, ax = rotv(m["ey"], m["ex"], sgr.rot, sgr.flip)
            psi = 0.5 * float(rng.vonmises(0.0, th.kappa)) if th.kappa > 1e-6 else float(rng.uniform(-math.pi / 2, math.pi / 2))
            dy, dx = rotv(ay, ax, psi)
            uy, ux = tp["u0"]
            c.phi_d = math.atan2(dy, dx) - math.atan2(uy, ux)
            c.psi, c.axis = psi, (ay, ax)
            sgm, im = plan[E - fbp]
            c.ecc_ref = pool.mom(sgm.L, sgm.frames[im])["ecc"]
            mit = [tp["E"] - fbp + 1 + i for i in range(fbp)]
            sg = Seg("mit", tp["M"], mit, c.phi_d, False, pool, exact_upto=fbp)
            for i in range(fbp):
                plan[E - fbp + 1 + i] = (sg, i)
            t = E + 1
        assert t == end + 1, (t, end, p0, c.start, c.nb)
        c.plan = plan

    def plan_step(self, c, t):
        dy, dx = self._plan_step(c, t)
        return dy * self.v_gain, dx * self.v_gain

    def _plan_step(self, c, t):
        if self.args.motion == "ou":
            return self.ou_step(c)
        sg, i = c.plan[t]
        prev = c.plan.get(t - 1)
        L, f = sg.L, sg.frames[i]
        fr = self.pool.T[L]["fr"]
        if prev is not None and prev[0] is sg and prev[1] == i - 1:
            pf = sg.frames[i - 1]
        elif (f - 1) in fr:
            pf = f - 1
        else:
            return self.ou_step(c)
        return rotv(fr[f][0] - fr[pf][0], fr[f][1] - fr[pf][1], sg.rot, sg.flip)

    def ou_step(self, c):
        th = self.th
        e = self.rng.standard_normal(2)
        c.v = th.phi * c.v + th.ou_s * math.sqrt(max(1 - th.phi ** 2, 0.0)) * e
        return float(c.v[0] + c.u[0]), float(c.v[1] + c.u[1])

    # ------------------------------------------------------------------ cell creation
    def new_cell(self, B, T, start, parent=0):
        c = Cell(self.next_wid)
        self.next_wid += 1
        c.B, c.T, c.start, c.parent = int(B), int(T), int(start), parent
        c.phi = float(self.rng.uniform(-math.pi, math.pi))
        c.flip = bool(self.rng.integers(2))
        c.z = float(self.rng.random())
        c.v = np.zeros(2)
        c.u = self.th.drift * self.rng.standard_normal(2)
        self.cells[c.wid] = c
        return c

    def schedule(self, c, lo):
        """division birth frame nb = B + T (phase-locked), own template, end, plan."""
        nb = c.B + c.T
        nb2 = self.lock(nb, lo)
        if nb2 != nb:
            c.T += nb2 - nb
        c.nb = c.B + c.T
        if c.nb <= self.NF - 1:
            c.div_tp = self.pool.divs[int(self.rng.integers(len(self.pool.divs)))]
            c.end = c.nb - 1
        else:
            c.end = self.NF - 1
        self.build_plan(c)

    # ------------------------------------------------------------------ initial cohort
    def init_world(self):
        th, rng, a = self.th, self.rng, self.args
        H, W = self.H, self.W
        cpf = th.cells_per_frame or self.pool_cpf
        self.n_canvas_target = max(1, int(round(a.density_scale * cpf * (H * W) / float(self.pool_lib_HW[0] * self.pool_lib_HW[1])
                                                * rng.uniform(0.8, 1.2))))
        self.h0, self.h, self.n_equiv = self.hazard(self.n_canvas_target)
        # --density_scale defines the expected CLIP-MEAN cells/frame: start lower by the expected growth over half the run
        self.n_canvas_start = max(1, int(round(self.n_canvas_target * math.exp(-self.h * (self.NF - 1) / 2.0))))
        self.mu_h = euler_lotka_mu(max(self.h, 1e-6), th.sigma)
        k = self.NF - 1
        self.pad = int(math.ceil(3 * math.sqrt(th.msd_pred(max(k, 1))) + 2 * th.D + 10))
        P = self.pad
        self.world = (-P, -P, H + P, W + P)
        n_world = int(round(self.n_canvas_start * (H + 2 * P) * (W + 2 * P) / float(H * W)))
        h = max(self.h, 1e-6)
        m = max(8 * n_world, 2000)
        Ts = np.maximum(np.exp(self.mu_h + th.sigma * rng.standard_normal(m)), th.Tmin)
        w = 1 - np.exp(-h * Ts)
        Tc = rng.choice(Ts, size=n_world, p=w / w.sum())
        U = rng.random(n_world)
        ages = -np.log(1 - U * (1 - np.exp(-h * Tc))) / h
        Ti = np.maximum(np.round(Tc).astype(int), th.Tmin)
        ai = np.minimum(np.floor(ages).astype(int), Ti - 1)
        # units: young cells (age < K_j) become sister pairs born before the clip
        order = rng.permutation(n_world)
        young = [i for i in order if ai[i] < th.Kj]
        old = [i for i in order if ai[i] >= th.Kj]
        units = []
        for j in range(0, len(young) - 1, 2):
            units.append(("pair", young[j], young[j + 1]))
        if len(young) % 2:
            old.append(young[-1])
        units += [("single", i, None) for i in old]
        units = [units[i] for i in rng.permutation(len(units))]
        placed_P, placed_r = [], []
        self.n_place_fail = 0
        for kind, i, j in units:
            if kind == "single":
                c = self.new_cell(-ai[i], Ti[i], 0)
                c.M = float(rng.choice(self.pool.M_pool))
                self.schedule(c, lo=1)
                cand = self._place([c], None, placed_P, placed_r)
                c.pos[0] = cand[0]
            else:
                age = int(ai[i])
                tp = self.pool.divs[int(rng.integers(len(self.pool.divs)))]
                phi_b = float(rng.uniform(-math.pi, math.pi))
                M = float(rng.choice(self.pool.M_pool))
                A_m = M * th.g(0.9) * th.rf[0]
                a1, a2 = self.pool.area(tp["d"][0], tp["E"] + 1), self.pool.area(tp["d"][1], tp["E"] + 1)
                cs = []
                # sister 2 shares the birth time: T2 ~ g_h | T2 > age (independent of sister 1's cycle)
                T2 = None
                for _ in range(200):
                    x = int(round(math.exp(self.mu_h + th.sigma * rng.standard_normal())))
                    if x > age and x >= th.Tmin:
                        T2 = x
                        break
                T2 = T2 if T2 is not None else max(age + 1, th.Tmin)
                for slot, (Tq, aq) in enumerate(((Ti[i], a1), (T2, a2))):
                    c = self.new_cell(-age, Tq, 0)
                    c.M = M * math.exp(0.05 * rng.standard_normal())
                    c.A_birth = th.dau_ratio * A_m * 2 * aq / float(a1 + a2)
                    c.birth_tp, c.birth_slot, c.phi_b = tp, slot, phi_b
                    c.init_young = True
                    self.schedule(c, lo=1)
                    cs.append(c)
                pair = self._new_pair(cs[0], cs[1], tp, phi_b, B=-age)
                rel = pair_rel(self, pair, age)
                cand = self._place(cs, rel, placed_P, placed_r)
                cs[0].pos[0], cs[1].pos[0] = cand[0], cand[1]
                pair["mid"] = np.array([(cand[0][0] + cand[1][0]) / 2, (cand[0][1] + cand[1][1]) / 2])
                self.pairs.append(pair)
        # pin the frame-0 canvas count (centroids inside) to n_canvas_start: drop random whole units (a single cell or a
        # young sister pair) inside the canvas, so the age mix (and the division rate) is unchanged
        ins = lambda c: 0 <= c.pos[0][0] < H and 0 <= c.pos[0][1] < W
        inside = [c for c in self.cells.values() if ins(c)]
        surplus = len(inside) - self.n_canvas_start
        pair_of = {}
        for p in self.pairs:
            pair_of[p["c"][0].wid] = p
            pair_of[p["c"][1].wid] = p
        removed = set()
        for k in rng.permutation(len(inside)):
            if surplus <= 0:
                break
            c = inside[int(k)]
            if c.wid in removed:
                continue
            grp = list(pair_of[c.wid]["c"]) if c.wid in pair_of else [c]
            for g in grp:
                removed.add(g.wid)
                del self.cells[g.wid]
            surplus -= sum(1 for g in grp if ins(g))
        self.pairs = [p for p in self.pairs if p["c"][0].wid not in removed]
        self.n_pinned = len(removed)
        self.alive = [c for c in self.cells.values()]

    def _place(self, cs, rel, PP, RR):
        """sequential placement with hard core c (r_i + r_j) and adjacency clustering p_adj."""
        rng = self.rng
        y0, x0, y1, x1 = self.world
        rs = [math.sqrt(self.area_target(c, 0) / math.pi) for c in cs]
        Pa = np.asarray(PP, float).reshape(-1, 2)
        Ra = np.asarray(RR, float)
        best, best_v = None, None
        cmin = max(self.c_ex, 0.9)
        for att in range(25):
            if len(PP) and rng.random() < self.p_adj:
                j = int(rng.integers(len(PP)))
                th_ = rng.uniform(0, 2 * math.pi)
                d = (rs[0] + Ra[j]) * rng.uniform(cmin, cmin + 0.25)
                cy, cx = Pa[j, 0] + d * math.sin(th_), Pa[j, 1] + d * math.cos(th_)
            else:
                cy, cx = rng.uniform(y0 + 5, y1 - 5), rng.uniform(x0 + 5, x1 - 5)
            if rel is None:
                pts = [(cy, cx)]
            else:
                # first sister at the anchor point, second at -rel (pair kept as drawn)
                pts = [(cy, cx), (cy - rel[0], cx - rel[1])]
            viol = 0.0
            if len(PP):
                for (py, px), r in zip(pts, rs):
                    d = np.hypot(Pa[:, 0] - py, Pa[:, 1] - px)
                    viol += float(np.maximum(self.c_ex * (Ra + r) - d, 0).sum())
            if best_v is None or viol < best_v:
                best, best_v = pts, viol
            if viol <= 0:
                break
        if best_v and best_v > 0:
            self.n_place_fail += 1
        for p, r in zip(best, rs):
            PP.append(p)
            RR.append(r)
        return best

    def _new_pair(self, c1, c2, tp, phi, B):
        c1.children, c2.children = [], []
        return dict(c=(c1, c2), tp=tp, phi=phi, B=B, mid=None, last_dir=None, active=True)

    # ------------------------------------------------------------------ division
    def divide(self, m, t):
        th, rng, pool = self.th, self.rng, self.pool
        E = t - 1
        tp, phi = m.div_tp, m.phi_d
        xm = m.pos[E]
        oy, ox = rotv(tp["off"][0], tp["off"][1], phi)
        A_m = self.area_target(m, E)
        a1, a2 = pool.area(tp["d"][0], tp["E"] + 1), pool.area(tp["d"][1], tp["E"] + 1)
        ds = []
        for slot, aq in enumerate((a1, a2)):
            d = self.new_cell(t, self.sample_T_newborn(), t, parent=m.wid)
            d.M = m.M * math.exp(0.05 * rng.standard_normal())
            d.A_birth = th.dau_ratio * A_m * 2 * aq / float(a1 + a2)
            d.birth_tp, d.birth_slot, d.phi_b = tp, slot, phi
            d.z = min(max(m.z + 0.02 * rng.standard_normal(), 0.0), 1.0)
            self.schedule(d, lo=t + th.Tmin)
            ds.append(d)
        m.children = [d.wid for d in ds]
        pair = self._new_pair(ds[0], ds[1], tp, phi, B=t)
        pair["mid"] = np.array([xm[0] + oy, xm[1] + ox])
        rel = pair_rel(self, pair, 0)
        ds[0].pos[t] = (pair["mid"][0] + rel[0] / 2, pair["mid"][1] + rel[1] / 2)
        ds[1].pos[t] = (pair["mid"][0] - rel[0] / 2, pair["mid"][1] - rel[1] / 2)
        self.pairs.append(pair)
        self.events_native.append(dict(mother=m.wid, daughters=[ds[0].wid, ds[1].wid], nb=t, psi=m.psi,
                                       axis=m.axis, phi_d=phi, tp_mother=int(tp["M"])))
        return ds

    # ------------------------------------------------------------------ world simulation
    def run(self):
        th, rng = self.th, self.rng
        self.init_world()
        alive = {c.wid: c for c in self.alive}
        y0, x0, y1, x1 = self.world
        pairs0 = {}
        for p in self.pairs:
            pairs0[p["c"][0].wid] = p
            pairs0[p["c"][1].wid] = p
        self._exclude(list(alive.values()), 0, pairs0)
        for p in self.pairs:
            c1, c2 = p["c"]
            p["mid"] = np.array([(c1.pos[0][0] + c2.pos[0][0]) / 2, (c1.pos[0][1] + c2.pos[0][1]) / 2])
        for t in range(1, self.NF):
            born = []
            for c in list(alive.values()):
                if c.nb == t:
                    born += self.divide(c, t)
                    del alive[c.wid]
            for d in born:
                alive[d.wid] = d
            in_pair = {}
            for p in self.pairs:
                if not p["active"]:
                    continue
                c1, c2 = p["c"]
                k = t - p["B"]
                if c1.wid not in alive or c2.wid not in alive or k > th.Kj:
                    p["active"] = False
                    continue
                in_pair[c1.wid] = p
                in_pair[c2.wid] = p
            for c in alive.values():
                if c.start == t or c.wid in in_pair:
                    continue
                dy, dx = self.plan_step(c, t)
                py, px = c.pos[t - 1]
                c.pos[t] = (min(max(py + dy, y0), y1), min(max(px + dx, x0), x1))
            done = set()
            for wid, p in in_pair.items():
                if id(p) in done:
                    continue
                done.add(id(p))
                c1, c2 = p["c"]
                k = t - p["B"]
                if c1.start == t:  # birth frame: already placed by divide()
                    continue
                st = pair_mid_step(self, p, k)
                if st is not None:
                    st = (st[0] * self.v_gain, st[1] * self.v_gain)
                if st is None:
                    s1, s2 = self.plan_step(c1, t), self.plan_step(c2, t)
                    st = (0.5 * (s1[0] + s2[0]), 0.5 * (s1[1] + s2[1]))
                else:  # keep the cells' plan pointers in sync (motion from the pair)
                    pass
                p["mid"] = p["mid"] + np.asarray(st)
                rel = pair_rel(self, p, k)
                c1.pos[t] = (p["mid"][0] + rel[0] / 2, p["mid"][1] + rel[1] / 2)
                c2.pos[t] = (p["mid"][0] - rel[0] / 2, p["mid"][1] - rel[1] / 2)
            self._exclude(list(alive.values()), t, in_pair)
            for wid, p in in_pair.items():
                c1, c2 = p["c"]
                p["mid"] = np.array([(c1.pos[t][0] + c2.pos[t][0]) / 2, (c1.pos[t][1] + c2.pos[t][1]) / 2])
        self.alive_end = alive

    def _exclude(self, cs, t, in_pair=None, iters=6):
        if len(cs) < 2:
            return
        P = np.array([c.pos[t] for c in cs], float)
        r = np.array([math.sqrt(max(self.area_target(c, t), 1.0) / math.pi) for c in cs])
        idx = {c.wid: i for i, c in enumerate(cs)}
        ex_pairs = set()
        if in_pair:
            for p in {id(p): p for p in in_pair.values()}.values():
                a, b = idx[p["c"][0].wid], idx[p["c"][1].wid]
                ex_pairs.add((min(a, b), max(a, b)))
        rmax = float(r.max())
        for _ in range(iters):
            tree = cKDTree(P)
            pr = tree.query_pairs(max(self.c_ex, 0.5) * 2 * rmax, output_type="ndarray")
            if not len(pr):
                break
            i, j = pr[:, 0], pr[:, 1]
            if ex_pairs:
                keep = np.array([(min(a, b), max(a, b)) not in ex_pairs for a, b in zip(i, j)])
                i, j = i[keep], j[keep]
                if not len(i):
                    break
            dv = P[i] - P[j]
            d = np.hypot(dv[:, 0], dv[:, 1])
            dmin = self.c_ex * (r[i] + r[j])
            v = d < dmin
            if not v.any():
                break
            i, j, dv, d, dmin = i[v], j[v], dv[v], d[v], dmin[v]
            z = d < 1e-6
            if z.any():
                ang = self.rng.uniform(0, 2 * math.pi, z.sum())
                dv[z] = np.stack([np.sin(ang), np.cos(ang)], 1)
                d[z] = 1.0
            push = (0.5 * (dmin - d) / d)[:, None] * dv
            acc = np.zeros_like(P)
            np.add.at(acc, i, push)
            np.add.at(acc, j, -push)
            P = P + acc
        y0, x0, y1, x1 = self.world
        P[:, 0] = np.clip(P[:, 0], y0, y1)
        P[:, 1] = np.clip(P[:, 1], x0, x1)
        for c, p in zip(cs, P):
            c.pos[t] = (float(p[0]), float(p[1]))

    # ------------------------------------------------------------------ rendering
    def crop(self, c, t):
        key = (c.wid, t)
        v = self.crop_cache.get(key)
        if v is not None:
            return v
        y, x = c.pos[t]
        pool, th = self.pool, self.th
        sg, i = c.plan[t]
        L, f = sg.L, sg.frames[i]
        src = pool.T[L]["fr"][f][2]
        m = pool.mom(L, f)
        A_t = self.area_target(c, t)
        Sm = None
        sxy = 1.0
        if sg.kind == "mit":
            E = c.nb - 1
            k = E - t
            e0 = min(m["ecc"], 0.97)
            e1 = min(max(c.ecc_ref + th.ed[k], 0.05), 0.97)
            ratio = math.sqrt((1 - e1 * e1) / max(1 - e0 * e0, 1e-6))  # minor/major scale ratio
            sx, sy = 1.0 / math.sqrt(ratio), math.sqrt(ratio)
            ex_, ey_ = m["ex"], m["ey"]
            Q = np.array([[ex_, -ey_], [ey_, ex_]])  # columns: major, minor (x, y)
            Sm = Q @ np.diag([sx, sy]) @ Q.T
            s2 = A_t / max(m["area"], 1)
        elif i <= sg.exact_upto:
            s2 = A_t / max(m["area"], 1)
        else:
            s2 = A_t / math.exp(sg.alpha + sg.beta * f)
        s = math.sqrt(min(max(s2, 0.2), 5.0))
        A = crop_affine(sg.rot, sg.flip, s, Sm)
        v = warp_crop(src, A, y - math.floor(y), x - math.floor(x))
        self.crop_cache[key] = v
        return v

    def render(self):
        """visibility runs -> output tracks (lineage remap) -> painted label stack."""
        H, W, I, nfo = self.H, self.W, self.I, self.nf_out
        items = {}  # (wid, j) -> (crop, ty, tx, y, x, vis_area)
        vis = {}
        for c in self.cells.values():
            for j in range(nfo):
                t = j * I
                if t not in c.pos:
                    continue
                y, x = c.pos[t]
                R = 2.2 * math.sqrt(max(self.area_target(c, t), 1) / math.pi) + 6
                if y < -R or x < -R or y > H + R or x > W + R:
                    continue
                cr, ay, ax = self.crop(c, t)
                h, w = cr.shape
                ty, tx = int(math.floor(y)) - ay, int(math.floor(x)) - ax
                y0, x0, y1, x1 = max(ty, 0), max(tx, 0), min(ty + h, H), min(tx + w, W)
                if y1 <= y0 or x1 <= x0:
                    continue
                sub = cr[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
                a_vis, a_full = int(sub.sum()), int(cr.sum())
                if a_vis < 0.3 * a_full or a_vis < 4:
                    continue
                items[(c.wid, j)] = (cr, ty, tx, y, x, a_vis)
                vis.setdefault(c.wid, []).append(j)
        # runs
        runs = {}  # wid -> list of [j0, j1]
        for wid, js in vis.items():
            rr = [[js[0], js[0]]]
            for j in js[1:]:
                if j == rr[-1][1] + 1:
                    rr[-1][1] = j
                else:
                    rr.append([j, j])
            runs[wid] = rr
        # linking: child first run starts at ceil(B/I) and the mother's last run ends at floor(E/I)
        link = {}  # child wid -> mother wid (run-level: child's first run, mother's last run)
        for c in self.cells.values():
            if not c.parent or c.wid not in runs:
                continue
            m = self.cells[c.parent]
            if m.wid not in runs:
                continue
            jb = -(-c.B // I)
            je = (m.end) // I
            if runs[c.wid][0][0] == jb and runs[m.wid][-1][1] == je and jb == je + 1:
                link[c.wid] = m.wid
        kids = {}
        for cw, mw in link.items():
            kids.setdefault(mw, []).append(cw)
        merged = {}  # child wid -> mother wid (single visible child continues the mother's track)
        for mw, ch in kids.items():
            if len(ch) == 1:
                merged[ch[0]] = mw
        # output tracks: key (wid, run index)
        tracks = {}  # key -> dict(j0, j1, parent_key)
        for wid, rr in runs.items():
            for ri, (j0, j1) in enumerate(rr):
                tracks[(wid, ri)] = dict(j0=j0, j1=j1, P=None)
        owner = {}  # (wid, ri) -> canonical key (after merges)

        def canon(k):
            while k in owner:
                k = owner[k]
            return k

        # merges first (child first run joins mother's last run), processed in time order
        for cw in sorted(merged, key=lambda w: self.cells[w].B):
            mw = merged[cw]
            ck, mk = (cw, 0), (mw, len(runs[mw]) - 1)
            mk = canon(mk)
            tracks[mk]["j1"] = tracks[ck]["j1"]
            owner[ck] = mk
        for cw, mw in link.items():
            if cw in merged:
                continue
            ck, mk = (cw, 0), canon((mw, len(runs[mw]) - 1))
            tracks[ck]["P"] = mk
        live = {k: v for k, v in tracks.items() if k not in owner}
        # entries 0: keep only tracks present at frame 0 and their descendants
        if int(self.args.entries) == 0:
            keep = set(k for k, v in live.items() if v["j0"] == 0)
            changed = True
            while changed:
                changed = False
                for k, v in live.items():
                    if k not in keep and v["P"] is not None and v["P"] in keep:
                        keep.add(k)
                        changed = True
            live = {k: v for k, v in live.items() if k in keep}
        # verify mothers have exactly 2 children; otherwise drop the links (should not happen after merging)
        ch_count = {}
        for k, v in live.items():
            if v["P"] is not None:
                ch_count[v["P"]] = ch_count.get(v["P"], 0) + 1
        self.n_bad_links = 0
        for k, v in live.items():
            if v["P"] is not None and (ch_count.get(v["P"]) != 2 or v["P"] not in live):
                v["P"] = None
                self.n_bad_links += 1
        # ids in order of first frame
        keys = sorted(live, key=lambda k: (live[k]["j0"], k[0], k[1]))
        oid = {k: i + 1 for i, k in enumerate(keys)}
        lineage = {}
        for k in keys:
            v = live[k]
            lineage[oid[k]] = (v["j0"], v["j1"], oid[v["P"]] if v["P"] is not None else 0)
        # frame -> list of (oid, item)
        frame_items = [[] for _ in range(nfo)]
        for (wid, j), it in items.items():
            rr = runs[wid]
            ri = next(q for q, (a, b) in enumerate(rr) if a <= j <= b)
            k = canon((wid, ri))
            if k not in oid:
                continue
            frame_items[j].append((oid[k], wid, it))
        self.lineage, self.oid, self.live, self.runs_vis = lineage, oid, live, runs
        self.key2oid = {k: oid[canon(k)] for k in tracks if canon(k) in oid}
        self.n_merged = len(merged)
        lab = np.zeros((nfo, H, W), np.uint16)
        self.paint_stats = dict(resolved_px=0, lost_fragment_px=0, occluded_px=0, cap_restored_cells=0,
                                forced_presence=0, max_loss_frac=0.0, loss_fracs=[])
        for j in range(nfo):
            lab[j] = self.paint(frame_items[j])
        self.lab = lab
        return lab

    def paint(self, its):
        H, W = self.H, self.W
        lab = np.zeros((H, W), np.int32)
        st = self.paint_stats
        cen = {}
        own_vis = {}
        if self.occ:
            its = sorted(its, key=lambda q: self.cells[q[1]].z)
        dist = np.full((H, W), np.inf, np.float32)
        subs = {}
        for oid, wid, (cr, ty, tx, y, x, a_vis) in its:
            h, w = cr.shape
            y0, x0, y1, x1 = max(ty, 0), max(tx, 0), min(ty + h, H), min(tx + w, W)
            sub = cr[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
            sl = (slice(y0, y1), slice(x0, x1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d2 = ((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32)
            region, dreg = lab[sl], dist[sl]
            occ = sub & (region > 0)
            if self.occ:
                take = sub
                st["occluded_px"] += int(occ.sum())
            else:
                take = sub & (d2 < dreg)
                st["resolved_px"] += int(occ.sum())
            region[take] = oid
            dreg[take] = d2[take]
            cen[oid] = (y, x)
            own_vis[oid] = int(sub.sum())
            subs[oid] = (sl, sub, d2)
        if self.occ:  # cap: the lower cell keeps >= 1 - F of its visible area (contested pixels by nearer centroid)
            areas = np.bincount(lab.ravel(), minlength=max(cen, default=0) + 1)
            for oid, wid, it in its:
                if areas[oid] < (1 - self.F) * own_vis[oid] - 0.5:
                    sl, sub, d2 = subs[oid]
                    region = lab[sl]
                    other = sub & (region > 0) & (region != oid)
                    yy, xx = np.nonzero(other)
                    if len(yy):
                        owners = region[yy, xx]
                        oc = np.array([cen[int(o)] for o in owners])
                        gy, gx = yy + sl[0].start, xx + sl[1].start
                        d_own = (gy - oc[:, 0]) ** 2 + (gx - oc[:, 1]) ** 2
                        win = d2[yy, xx] < d_own
                        region[yy[win], xx[win]] = oid
                    st["cap_restored_cells"] += 1
        # one 4-connected component per id
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs):
            if sl is None:
                continue
            oid = i + 1
            m = lab[sl] == oid
            cc, k = ndi.label(m)
            if k > 1:
                sizes = ndi.sum(m, cc, range(1, k + 1))
                keep = 1 + int(np.argmax(sizes))
                lab[sl][m & (cc != keep)] = 0
                st["lost_fragment_px"] += int(sizes.sum() - sizes.max())
        # presence: every listed id must keep pixels (force a core disc if not)
        areas = np.bincount(lab.ravel(), minlength=max(cen, default=0) + 1)
        for oid, wid, it in its:
            if areas[oid] < 4:
                sl, sub, d2 = subs[oid]
                core = sub & (d2 <= max(4.0, np.sort(d2[sub])[min(5, int(sub.sum()) - 1)]))
                lab[sl][core] = oid
                st["forced_presence"] += 1
        # re-check components after forced presence
        if st["forced_presence"]:
            objs = ndi.find_objects(lab)
            for i, sl in enumerate(objs):
                if sl is None:
                    continue
                m = lab[sl] == i + 1
                cc, k = ndi.label(m)
                if k > 1:
                    sizes = ndi.sum(m, cc, range(1, k + 1))
                    lab[sl][m & (cc != 1 + int(np.argmax(sizes)))] = 0
        areas = np.bincount(lab.ravel(), minlength=max(cen, default=0) + 1)
        for oid in own_vis:
            lf = 1 - areas[oid] / max(own_vis[oid], 1)
            st["loss_fracs"].append(lf)
            st["max_loss_frac"] = max(st["max_loss_frac"], float(lf))
        return lab.astype(np.uint16)


def pair_rel(sim, p, k):
    """sister relative vector (y, x) at age k.  While the real template pair is tracked: its own relative vector,
    rotated onto the drawn Hertwig axis (the real per-frame jitter is kept; theta's d(k) is the median of exactly these
    pairs, so the separation follows d(k) in distribution).  Afterwards: |r| = rho_pair D d(k) along the last
    direction with a small angular random walk (rho_pair = the pair's mean ratio to d(k))."""
    th, pool = sim.th, sim.pool
    tp, phi = p["tp"], p["phi"]
    t = tp["E"] + 1 + k
    d1, d2 = tp["d"]
    f1, f2 = tp["f"]
    if t <= f1[-1] and t <= f2[-1]:
        a, b = pool.T[d1]["fr"][t], pool.T[d2]["fr"][t]
        uy, ux = rotv(a[0] - b[0], a[1] - b[1], phi)
        n = math.hypot(uy, ux)
        if n > 1e-6:
            p["last_dir"] = (uy / n, ux / n)
            return uy, ux
    if p.get("last_dir") is None:
        uy, ux = rotv(tp["u0"][0], tp["u0"][1], phi)
        n = math.hypot(uy, ux)
        p["last_dir"] = (uy / n, ux / n)
    else:
        a = 0.08 * sim.rng.standard_normal()
        p["last_dir"] = rotv(p["last_dir"][0], p["last_dir"][1], a)
    mag = tp["rho"] * th.D * th.dsis(k)
    return mag * p["last_dir"][0], mag * p["last_dir"][1]


def pair_mid_step(sim, p, k):
    tp, phi = p["tp"], p["phi"]
    t = tp["E"] + 1 + k
    d1, d2 = tp["d"]
    f1, f2 = tp["f"]
    if k >= 1 and t <= f1[-1] and t <= f2[-1]:
        T = sim.pool.T
        a1, b1 = T[d1]["fr"][t], T[d2]["fr"][t]
        a0, b0 = T[d1]["fr"][t - 1], T[d2]["fr"][t - 1]
        dy = 0.5 * (a1[0] + b1[0]) - 0.5 * (a0[0] + b0[0])
        dx = 0.5 * (a1[1] + b1[1]) - 0.5 * (a0[1] + b0[1])
        return rotv(dy, dx, phi)
    return None


# ============================================================================ clip
def frame_stats(lab):
    """touching fraction and NN distances (kinstats definitions) of one label image."""
    ids = np.unique(lab)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None, []
    touch = touching_labels(lab)
    cm = ndi.center_of_mass(np.ones_like(lab), lab, ids)
    pts = np.asarray(cm, float)
    nn = []
    if len(pts) > 1:
        dd, _ = cKDTree(pts).query(pts, k=2)
        nn = dd[:, 1].tolist()
    return len(touch) / len(ids), nn


def verify_stack(lab, lineage):
    nf = lab.shape[0]
    present = {}
    one_cc4 = one_cc8 = 0
    cc_list = []
    for t in range(nf):
        for i, sl in enumerate(ndi.find_objects(lab[t])):
            if sl is None:
                continue
            lid = i + 1
            present.setdefault(lid, []).append(t)
            m = lab[t][sl] == lid
            k4 = ndi.label(m)[1]
            if k4 != 1:
                one_cc4 += 1
                cc_list.append((lid, t, int(k4)))
            if ndi.label(m, structure=S8)[1] != 1:
                one_cc8 += 1
    bad = 0
    nch = {}
    for lid, (B, E, P) in lineage.items():
        if present.get(lid, []) != list(range(B, E + 1)):
            bad += 1
        if P > 0:
            nch[P] = nch.get(P, 0) + 1
            if P not in lineage or lineage[P][1] + 1 != B:
                bad += 1
    bad += sum(1 for v in nch.values() if v != 2)
    if set(present) != set(lineage):
        bad += 1
    return one_cc4, one_cc8, bad, cc_list


def simulate(pool, th, args, seed, calib):
    sim = Sim(pool, th, args, seed, calib)
    sim.pool_lib_HW = POOL_HW
    sim.pool_cpf = POOL_CPF
    sim.run()
    sim.render()
    return sim


def make_clip(pool, th, args, calib, clip_idx, out_dir, ctc_dir):
    t0 = time.time()
    seed = args.seed * 100000 + clip_idx
    sim = simulate(pool, th, args, seed, calib)
    lab, lineage = sim.lab, sim.lineage
    H, W, nf, I = sim.H, sim.W, sim.nf_out, sim.I
    one_cc4, one_cc8, lineage_bad, cc_list = verify_stack(lab, lineage)
    mothers, daughters = CR.division_events(lineage)
    births = sorted(set(daughters.values()))
    tb_list = [lineage[m][1] + 1 for m in mothers]
    lock_viol = sum(1 for tb in tb_list if sim.allowed and ((tb - 1) % 4) not in sim.allowed)
    entries = [L for L, (B, E, P) in lineage.items() if P == 0 and B > 0]
    exits = [L for L, (B, E, P) in lineage.items() if L not in mothers and E < nf - 1]
    cells = [int(len(np.unique(lab[t])) - 1) for t in range(nf)]
    cpairs = [LS.contact_pairs(lab[t]) for t in range(nf)]
    tf, nnd = [], []
    for t in range(nf):
        a, b = frame_stats(lab[t])
        if a is not None:
            tf.append(a)
        nnd += b
    atrisk = float(sum(cells[:-1]))
    # ---- write (lineage_sim.py layout)
    cid = f"clip_{clip_idx:03d}"
    cdir = os.path.join(out_dir, cid)
    cond_dir = CR.cond_dirname(args.cond_version)
    for sub in ("labels", cond_dir):
        os.makedirs(os.path.join(cdir, sub), exist_ok=True)
    tra_dir = os.path.join(ctc_dir, f"{cid}_GT", "TRA")
    seg_dir = os.path.join(ctc_dir, f"{cid}_ST", "SEG")
    os.makedirs(tra_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    rad = LS.MARKER_R[args.dataset]
    for t in range(nf):
        L = lab[t]
        cond = CR.cond_image(L, lineage, t, args.cond_version, mothers, daughters)
        cv2.imwrite(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), L, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(cdir, cond_dir, f"frame_{t:03d}.png"), cond[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tifffile.imwrite(os.path.join(seg_dir, f"man_seg{t:03d}.tif"), L, compression="zlib")
        tifffile.imwrite(os.path.join(tra_dir, f"man_track{t:03d}.tif"), LS.markers_image(L, rad), compression="zlib")
    lines = [f"{lid} {B} {E} {P}\n" for lid, (B, E, P) in sorted(lineage.items())]
    with open(os.path.join(cdir, "lineage.txt"), "w") as f:
        f.writelines(lines)
    with open(os.path.join(tra_dir, "man_track.txt"), "w") as f:
        f.writelines(lines)
    ref, ref_info = LS.load_ref(args.dataset, "02", args.root, sim.rng, H, W)
    cv2.imwrite(os.path.join(cdir, "ref.png"), ref[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    # ---- events (labelled divisions with their biology)
    ev = []
    for e in sim.events_native:
        rr = sim.runs_vis.get(e["mother"])
        if not rr:
            continue
        m_o = sim.key2oid.get((e["mother"], len(rr) - 1))
        if m_o is None or m_o not in mothers or lineage[m_o][1] + 1 != -(-e["nb"] // I):
            continue
        ch = sorted(L for L, (B, E, P) in lineage.items() if P == m_o)
        ev.append(dict(mother=m_o, daughters=ch, t_birth=lineage[m_o][1] + 1, native_birth=e["nb"],
                       psi_deg=round(math.degrees(e["psi"]), 2), axis_yx=[round(v, 4) for v in e["axis"]],
                       template_mother=e["tp_mother"]))
    lf = np.asarray(sim.paint_stats["loss_fracs"], float)
    events = dict(divisions=ev, entries=sorted(entries), exits=sorted(exits),
                  phase_lock=sorted(sim.allowed), lock_shifts_native=sim.lock_shifts)
    with open(os.path.join(cdir, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    meta = dict(dataset=args.dataset, video="02", clip=cid, seed=seed, num_frames=nf, height=H, width=W, synthetic=True,
                generator=f"lineage_sim_v3.py {VERSION}", cond_version=int(args.cond_version), cond_dir=cond_dir,
                root=args.root, theta=th.path, theta_sha=th.sha, library=args.cache,
                params=dict(density_scale=args.density_scale, event_budget=args.event_budget, interval=I,
                            phase_lock=args.phase_lock, overlap_mode=args.overlap_mode, overlap_max=args.overlap_max,
                            entries=int(args.entries), motion=args.motion),
                calibration=dict(c_exclusion=sim.c_ex, p_adj=sim.p_adj, hazard_gain=sim.h_gain, speed_gain=sim.v_gain,
                                 source=calib.get("source")),
                kinetics=dict(hazard_fitted=sim.h0, hazard=sim.h, n_equiv_native=sim.n_equiv, mu_h_initial=sim.mu_h,
                              mu_newborn=th.mu - math.log(max(float(args.event_budget), 1e-3)), sigma=th.sigma,
                              kappa=th.kappa, K_joint=th.Kj, Tmin=th.Tmin, rounding=th.round_src, pad_px=sim.pad,
                              native_frames=sim.NF, n_world_cells=len(sim.cells), n_canvas_target=sim.n_canvas_target,
                              n_canvas_start=sim.n_canvas_start, n_pinned_removed=sim.n_pinned),
                n_div=len(mothers), division_frames=sorted(tb_list), n_ids=len(lineage), cells_per_frame=cells,
                mean_cells_per_frame=float(np.mean(cells)), contact_pairs_per_frame=cpairs,
                mean_contact_pairs_per_frame=float(np.mean(cpairs)),
                div_per_1000_atrisk=1000.0 * len(mothers) / max(atrisk, 1), n_entries=len(entries), n_exits=len(exits),
                entry_per_1000_atrisk=1000.0 * len(entries) / max(atrisk, 1),
                exit_per_1000_atrisk=1000.0 * len(exits) / max(atrisk, 1),
                touching_fraction=float(np.mean(tf)) if tf else None,
                nn_px_q10_50_90=[float(np.percentile(nnd, q)) for q in (10, 50, 90)] if nnd else None,
                merged_single_child=sim.n_merged, dropped_links=sim.n_bad_links, place_fail=sim.n_place_fail,
                resolved_overlap_px=sim.paint_stats["resolved_px"], occluded_px=sim.paint_stats["occluded_px"],
                lost_fragment_px=sim.paint_stats["lost_fragment_px"], cap_restored=sim.paint_stats["cap_restored_cells"],
                forced_presence=sim.paint_stats["forced_presence"],
                visible_area_loss_frac=dict(median=float(np.median(lf)) if len(lf) else 0.0,
                                            q99=float(np.percentile(lf, 99)) if len(lf) else 0.0,
                                            max=float(lf.max()) if len(lf) else 0.0),
                overlap_violations=0, violations_one_cc=one_cc4, violations_one_cc_8conn=one_cc8,
                violations_one_cc_list=cc_list[:50], lineage_violations=lineage_bad, phase_lock_violations=lock_viol,
                ref=ref_info, seconds=time.time() - t0)
    with open(os.path.join(cdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ============================================================================ calibration (e)
def link_errors(lab, gaps=(1, 4, 8)):
    """kinstats s12 on a label stack: share of cells (same id at t and t+g) whose nearest cell at t+g is another cell."""
    nf = lab.shape[0]
    cents = []
    for t in range(nf):
        ids = np.unique(lab[t])
        ids = ids[ids > 0]
        cm = np.asarray(ndi.center_of_mass(np.ones_like(lab[t]), lab[t], ids), float).reshape(-1, 2)
        cents.append((ids, cm))
    out = {}
    for g in gaps:
        err = []
        for t in range(nf - g):
            ids0, c0 = cents[t]
            ids1, c1 = cents[t + g]
            if len(ids1) < 2 or not len(ids0):
                continue
            idx1 = {int(i): k for k, i in enumerate(ids1)}
            com = [k for k, i in enumerate(ids0) if int(i) in idx1]
            if not com:
                continue
            q = c0[com]
            dd, ii = cKDTree(c1).query(q, k=2)
            tr = np.array([idx1[int(ids0[k])] for k in com])
            dtrue = np.hypot(*(c1[tr] - q).T)
            dalt = np.where(ii[:, 0] == tr, dd[:, 1], dd[:, 0])
            err += (dalt < dtrue).tolist()
        out[g] = (float(np.sum(err)), len(err))
    return out


def _calib_job(job):
    c, p, seeds, args_d = job
    a = argparse.Namespace(**args_d)
    tfs, nns, ent, ext, ar, ndiv = [], [], 0, 0, 0.0, 0
    le = {1: [0.0, 0], 4: [0.0, 0], 8: [0.0, 0]}
    for s in seeds:
        sim = simulate(POOL, THETA, a, s, dict(c=c, p_adj=p))
        lab = sim.lab
        for t in range(lab.shape[0]):
            x, nn = frame_stats(lab[t])
            if x is not None:
                tfs.append(x)
            nns += nn
        lin = sim.lineage
        mothers, _ = CR.division_events(lin)
        nf = lab.shape[0]
        ent += sum(1 for L, (B, E, P) in lin.items() if P == 0 and B > 0)
        ext += sum(1 for L, (B, E, P) in lin.items() if L not in mothers and E < nf - 1)
        ar += sum(len(np.unique(lab[t])) - 1 for t in range(nf - 1))
        ndiv += len(mothers)
        for g, (e_, n_) in link_errors(lab).items():
            le[g][0] += e_
            le[g][1] += n_
    q = [float(np.percentile(nns, x)) for x in (10, 50, 90)] if nns else [0, 0, 0]
    return dict(c=c, p_adj=p, touch=float(np.mean(tfs)), nn_q=q, entry=1000 * ent / max(ar, 1), n_div=ndiv, atrisk=ar,
                exit=1000 * ext / max(ar, 1), div=1000 * ndiv / max(ar, 1),
                link=[le[g][0] / max(le[g][1], 1) for g in (1, 4, 8)])


def speed_steps(lab):
    """kinstats s07 on a label stack: centroid step (px/frame) of every id present in consecutive frames."""
    out = []
    prev = None
    for t in range(lab.shape[0]):
        ids = np.unique(lab[t])
        ids = ids[ids > 0]
        cm = np.asarray(ndi.center_of_mass(np.ones_like(lab[t]), lab[t], ids), float).reshape(-1, 2)
        cur = {int(i): c for i, c in zip(ids, cm)}
        if prev is not None:
            out += [math.hypot(c[0] - prev[i][0], c[1] - prev[i][1]) for i, c in cur.items() if i in prev]
        prev = cur
    return out


def _speed_job(job):
    c, p, hg, g, seeds, args_d = job
    a = argparse.Namespace(**args_d)
    sp = []
    for s in seeds:
        sim = simulate(POOL, THETA, a, s, dict(c=c, p_adj=p, hazard_gain=hg, speed_gain=g))
        sp += speed_steps(sim.lab)
    return [float(x) for x in np.quantile(np.asarray(sp, float), (np.arange(200) + 0.5) / 200.0)]


def run_speed_jobs(jobs, workers):
    if workers > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pl:
            return pl.map(_speed_job, jobs)
    return [_speed_job(j) for j in jobs]


def calibrate(args, th, workers):
    """grid over (c, p_adj) at the source density (density_scale 1, interval 1, 17 frames): touching fraction and NN
    quantiles vs theta; cached per (dataset, theta, library, overlap mode)."""
    key = dict(ds=args.dataset, theta=th.sha, lib=os.path.basename(args.cache), mode=args.overlap_mode,
               F=float(args.overlap_max) if args.overlap_mode == "occlude" else 0.0, v=VERSION, motion=args.motion,
               n=int(args.calib_clips), link_ref=args.calib_link_ref or "")
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    cdir = os.path.join(N, "eval", "mitosync", "calib")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"calib_v3_{args.dataset}_{args.overlap_mode}_{h}.json")
    if os.path.exists(path):
        r = json.load(open(path))
        log(f"  calibration loaded {path}: c={r['best']['c']} p_adj={r['best']['p_adj']} "
            f"(touch {r['best']['touch']:.3f} vs {th.touch:.3f}, NN {r['best']['nn_q']} vs {th.nn_q})")
        return dict(c=r["best"]["c"], p_adj=r["best"]["p_adj"], hazard_gain=r["best"].get("hazard_gain", 1.0),
                    speed_gain=r["best"].get("speed_gain", 1.0), source=path)
    d = dict(vars(args))
    d.update(density_scale=1.0, interval=1, num_frames=17, phase_lock="", entries=1)
    tq = np.asarray(th.nn_q)
    link_ref = None
    speed_ref = None
    if args.calib_link_ref and os.path.exists(args.calib_link_ref):
        R = json.load(open(args.calib_link_ref))
        link_ref = [R["stats12"]["s12_nn_link_error"][f"g{g}"] for g in (1, 4, 8)]
        if (R.get("quantiles") or {}).get("speed"):
            speed_ref = np.asarray(R["quantiles"]["speed"], float)  # kinstats 200-point speed quantiles of the HEAD

    def score(res):
        for r in res:
            r["J_touch"] = abs(r["touch"] - th.touch) / 0.05
            r["J_nn"] = float(np.mean(np.abs(np.asarray(r["nn_q"][:2]) - tq[:2])) / tq[1])
            r["J_link"] = float(np.mean([abs(math.log((a + 5e-4) / (b + 5e-4))) for a, b in zip(r["link"], link_ref)])) \
                if link_ref else 0.0
            r["J"] = r["J_touch"] + r["J_nn"] / 0.1 + r["J_link"] / 0.5
        return res

    def run_jobs(jobs):
        if workers > 1:
            from multiprocessing import Pool
            with Pool(min(workers, len(jobs))) as pl:
                return pl.map(_calib_job, jobs)
        return [_calib_job(j) for j in jobs]

    t0 = time.time()
    # stage 1: coarse grid, n clips per point
    grid = [(c, p) for c in (0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5) for p in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)]
    seeds = [900000 + i for i in range(int(args.calib_clips))]
    res1 = score(run_jobs([(c, p, seeds, d) for c, p in grid]))
    # stage 2: the 8 best points re-evaluated on 3x as many fresh clips (less noise), pick the best
    top = sorted(res1, key=lambda r: r["J"])[:8]
    seeds2 = [910000 + i for i in range(3 * int(args.calib_clips))]
    res = score(run_jobs([(r["c"], r["p_adj"], seeds2, d) for r in top]))
    best = min(res, key=lambda r: r["J"])
    # hazard gain: theta rate / labelled rate per at-risk cell-frame, pooled over all stage-2 runs (gain 1 inside)
    nd, ar = sum(r["n_div"] for r in res), sum(r["atrisk"] for r in res)
    gain = float(np.clip(th.rate / max(nd / max(ar, 1), 1e-9), 0.8, 1.4)) if nd >= 50 else 1.0
    best["hazard_gain"] = gain
    # v3.8 stage 3: motion gain at the chosen (c, p_adj, hazard_gain): rendered speed quantiles vs the HEAD (min W1)
    sg, sgrid = 1.0, []
    if speed_ref is not None:
        gains = [round(0.80 + 0.02 * i, 2) for i in range(21)]  # 0.80 .. 1.20
        seeds3 = [920000 + i for i in range(3 * int(args.calib_clips))]
        with_g = run_speed_jobs([(best["c"], best["p_adj"], gain, g, seeds3, d) for g in gains], workers)
        for g, q in zip(gains, with_g):
            w = float(np.mean(np.abs(np.asarray(q) - speed_ref)))
            sgrid.append(dict(gain=g, w1=w, q10_50_90=[q[19], q[99], q[179]]))
        sg = min(sgrid, key=lambda r: r["w1"])["gain"]
    best["speed_gain"] = sg
    out = dict(key=key, objective="|touch - theta|/0.05 + mean_q|NN_q - theta_q|/NN_median/0.1 (q = 10/50; q90 reported only: "
                         "it is set by the head's density spread, not by contact physics) + mean_g|ln((link_g+5e-4)/(ref_g+5e-4))|/0.5 "
                         "(NN linking error at gaps 1/4/8 of the HEAD 17-frame windows, kinstats s12, head only)",
               target=dict(touch=th.touch, nn_q=list(th.nn_q), entry=th.entry, exit=th.exit, div_per_1000=1000 * th.rate,
                           link_head_w17=link_ref, link_ref_file=args.calib_link_ref),
               grid=res1, refine=res, best=best, speed_gain_grid=sgrid,
               speed_ref=dict(file=args.calib_link_ref, q10_50_90=None if speed_ref is None else
                              [float(speed_ref[19]), float(speed_ref[99]), float(speed_ref[179])]),
               seconds=time.time() - t0,
               note="calibrated at the source density (density_scale 1, the head's cells/frame); the physical parameters "
                    "(c, p_adj) are reused at every requested density, so touching/NN scale with density as in real data")
    json.dump(out, open(path, "w"), indent=1)
    log(f"  calibration -> {path} ({out['seconds']:.0f}s): c={best['c']} p_adj={best['p_adj']} touch {best['touch']:.3f} "
        f"(theta {th.touch:.3f}) NN q {np.round(best['nn_q'], 1).tolist()} (theta {np.round(tq, 1).tolist()}) "
        f"entry/exit {best['entry']:.2f}/{best['exit']:.2f} (theta {th.entry:.2f}/{th.exit:.2f}) div {best['div']:.2f} "
        f"link {np.round(best['link'], 4).tolist()} (head {link_ref}) hazard_gain {gain:.3f}")
    log(f"  speed_gain {sg} (head speed q10/50/90 {None if speed_ref is None else np.round(speed_ref[[19, 99, 179]], 3).tolist()}; "
        f"grid {[(r['gain'], round(r['w1'], 4)) for r in sgrid]})")
    return dict(c=best["c"], p_adj=best["p_adj"], hazard_gain=gain, speed_gain=sg, source=path)


# ============================================================================ montage
def montage(out_dir, metas, fig_path, nf, scale=0.5, n_clips=2):
    frames = [0, nf // 2, nf - 1]
    rows = []
    for m in metas[:n_clips]:
        cdir = os.path.join(out_dir, m["clip"])
        ev = json.load(open(os.path.join(cdir, "events.json")))["divisions"]
        lin = CR.load_lineage(os.path.join(cdir, "lineage.txt"))
        labs = [cv2.imread(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), -1) for t in frames]
        conds = [cv2.imread(os.path.join(cdir, m["cond_dir"], f"frame_{t:03d}.png"), -1)[..., ::-1].copy() for t in frames]
        cols = [LS.colorize(L) for L in labs]
        for row_imgs, is_lab in ((conds, False), (cols, True)):
            ims = []
            for t, im, L in zip(frames, row_imgs, labs):
                im = np.ascontiguousarray(im)
                for e in ev:
                    tb = e["t_birth"]
                    if abs(tb - t) > 4:
                        continue
                    ids = e["daughters"] if tb <= t else [e["mother"]]
                    col = (255, 0, 255) if tb <= t else (255, 255, 0)
                    pts = []
                    for i in ids:
                        yy, xx = np.nonzero(L == i)
                        if len(yy):
                            pts.append((yy.mean(), xx.mean()))
                    if not pts:
                        continue
                    cy = float(np.mean([p[0] for p in pts]))
                    cx = float(np.mean([p[1] for p in pts]))
                    cv2.circle(im, (int(cx), int(cy)), 22, col, 2, cv2.LINE_AA)
                    cv2.putText(im, f"{tb}", (int(cx) + 18, int(cy) - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                ims.append(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if not is_lab else cv2.INTER_NEAREST))
            sep = np.full((ims[0].shape[0], 4, 3), 255, np.uint8)
            rows.append(np.concatenate([x for im in ims for x in (im, sep)][:-1], 1))
    img = np.concatenate([x for r in rows for x in (r, np.full((4, r.shape[1], 3), 255, np.uint8))][:-1], 0)
    y = 0
    rh = rows[0].shape[0]
    for i, m in enumerate(metas[:n_clips]):
        for name in ("cond_v2" if m["cond_version"] == 2 else "cond", "labels"):
            cv2.putText(img, f"{m['clip']} {name}: {m['mean_cells_per_frame']:.0f} cells/frame, {m['n_div']} divisions, "
                             f"{m['n_entries']} entries", (6, y + rh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += rh + 4
    w3 = (img.shape[1] + 8) // 3
    for j, t in enumerate(frames):
        cv2.putText(img, f"frame {t}", (j * w3 + w3 - 80, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    leg = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(leg, "yellow circle: mother dividing within 4 frames (number = birth frame t_b);  magenta: daughters born "
                     "within the last 4 frames", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.concatenate([img, leg], 0)
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    cv2.imwrite(fig_path, np.ascontiguousarray(img[..., ::-1]))


def event_strip(out_dir, metas, fig_path, n_events=4, half=4, box=None, up=3):
    """zoomed filmstrips of labelled divisions (labels colourised + cond), frames t_b-half .. t_b+half-1."""
    strips = []
    got = 0
    for m in metas:
        cdir = os.path.join(out_dir, m["clip"])
        ev = json.load(open(os.path.join(cdir, "events.json")))["divisions"]
        nf = m["num_frames"]
        for e in ev:
            tb = e["t_birth"]
            if tb - half < 0 or tb + half - 1 > nf - 1:
                continue
            labs = [cv2.imread(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), -1) for t in range(tb - half, tb + half)]
            conds = [cv2.imread(os.path.join(cdir, m["cond_dir"], f"frame_{t:03d}.png"), -1)[..., ::-1] for t in range(tb - half, tb + half)]
            L = labs[half]
            pts = [np.nonzero(L == d) for d in e["daughters"]]
            if any(len(p[0]) == 0 for p in pts):
                continue
            cy = int(np.mean([p[0].mean() for p in pts]))
            cx = int(np.mean([p[1].mean() for p in pts]))
            b = box or int(3.2 * math.sqrt(max(sum(len(p[0]) for p in pts), 1)))
            H, W = L.shape
            y0, x0 = min(max(cy - b // 2, 0), H - b), min(max(cx - b // 2, 0), W - b)
            row_l, row_c = [], []
            for L2, C2, t in zip(labs, conds, range(tb - half, tb + half)):
                sub = L2[y0:y0 + b, x0:x0 + b]
                col = np.zeros(sub.shape + (3,), np.uint8)
                col[sub > 0] = (90, 90, 90)
                col[sub == e["mother"]] = (255, 200, 0)
                col[sub == e["daughters"][0]] = (255, 0, 255)
                col[sub == e["daughters"][1]] = (0, 220, 255)
                col = cv2.resize(col, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
                cc = cv2.resize(np.ascontiguousarray(C2[y0:y0 + b, x0:x0 + b]), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
                cv2.putText(col, f"t{t}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                row_l.append(np.pad(col, ((2, 2), (2, 2), (0, 0)), constant_values=255))
                row_c.append(np.pad(cc, ((2, 2), (2, 2), (0, 0)), constant_values=255))
            r = np.concatenate([np.concatenate(row_l, 1), np.concatenate(row_c, 1)], 0)
            cv2.putText(r, f"{m['clip']} t_b={tb} psi={e['psi_deg']:.0f}deg", (4, r.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 0), 1, cv2.LINE_AA)
            strips.append(r)
            got += 1
            break
        if got >= n_events:
            break
    if not strips:
        return
    wmax = max(s.shape[1] for s in strips)
    strips = [np.pad(s, ((0, 6), (0, wmax - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
    img = np.concatenate(strips, 0)
    cv2.imwrite(fig_path, np.ascontiguousarray(img[..., ::-1]))


# ============================================================================ main
POOL = THETA = ARGS = CALIB = None
POOL_HW = POOL_CPF = None


def _job(j):
    return make_clip(POOL, THETA, ARGS, CALIB, *j)


def main():
    global POOL, THETA, ARGS, CALIB, POOL_HW, POOL_CPF
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LS.CANVAS))
    ap.add_argument("--theta", required=True)
    ap.add_argument("--root", default=f"{N}/data/mitosync/kin_head")
    ap.add_argument("--cache", required=True, help="head-only replay library pickle (lineage_sim.build_library format)")
    ap.add_argument("--video", default="02")
    ap.add_argument("--num_clips", type=int, default=20)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--density_scale", type=float, default=1.0)
    ap.add_argument("--event_budget", type=float, default=1.0, help="multiplies the fitted division rate")
    ap.add_argument("--interval", type=int, default=1, help="native frames per output frame (1 or 8)")
    ap.add_argument("--phase_lock", default="", help="allowed (t_b-1) mod 4 positions, e.g. '1,2'; empty = unconstrained")
    ap.add_argument("--overlap_mode", default="none", choices=["none", "occlude"])
    ap.add_argument("--overlap_max", type=float, default=0.15)
    ap.add_argument("--entries", type=int, default=1, choices=[0, 1])
    ap.add_argument("--cond_version", type=int, default=2, choices=[1, 2])
    ap.add_argument("--motion", default="replay", choices=["replay", "ou"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--calib", default="auto", help="'auto' (cached grid calibration) or 'c,p_adj[,hazard_gain[,speed_gain]]'")
    ap.add_argument("--speed_gain", default="auto", help="v3.8 motion gain: 'auto' (from the calibration) or a number "
                    "(1 = v3.7 behaviour)")
    ap.add_argument("--calib_clips", type=int, default=4)
    ap.add_argument("--calib_link_ref", default=None, help="head kinstats json for the link-error term (default: "
                    "$N/eval/mitosync/kinstats/<ds>__real_head_w17.json, head only; '' disables)")
    args = ap.parse_args()
    if args.video != "02" or "01" in args.root.split("/")[-1:]:
        sys.exit("REFUSING: only the 02 head may be used (video 01 is held out).")
    if args.phase_lock and not all(x.strip() in ("0", "1", "2", "3") for x in args.phase_lock.split(",")):
        sys.exit("--phase_lock must be a comma-separated subset of 0,1,2,3")
    if args.calib_link_ref is None:
        args.calib_link_ref = f"{N}/eval/mitosync/kinstats/{args.dataset}__real_head_w17.json"
    th = Theta(args.theta)
    if th.dataset != args.dataset:
        sys.exit(f"theta dataset {th.dataset} != {args.dataset}")
    if os.path.exists(args.cache):
        lib = pickle.load(open(args.cache, "rb"))
        log(f"  library {args.cache}: {len(lib['tracks'])} tracks, {len(lib['runs'])} runs")
    else:
        lib = LS.build_library(args.dataset, "02", args.root)
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        pickle.dump(lib, open(args.cache, "wb"), protocol=4)
    mx = max((max(tk["fr"]) for tk in lib["tracks"].values() if tk["fr"]), default=-1)
    a, b = th.raw["source_frames"]
    if mx > b:
        sys.exit(f"REFUSING: library has frame {mx} beyond the theta source window {a}-{b} (head only)")
    nref = len([f for f in os.listdir(f"{args.root}/{args.dataset}/02") if f.endswith(".tif")])
    if nref - 1 > b:
        sys.exit(f"REFUSING: ref root has {nref} frames > head window {a}-{b}")
    POOL = LibPool(lib, th)
    POOL_HW = (lib["H"], lib["W"])
    POOL_CPF = lib["real_cells_per_frame"]
    THETA, ARGS = th, args
    log(f"[{args.dataset}] theta {th.path} ({th.sha}); pool: {len(POOL.pieces)} interphase pieces, {len(POOL.divs)} division "
        f"templates; D={th.D:.1f}px rate={th.rate:.4f} beta={th.beta:.2f} mu={th.mu:.3f} sigma={th.sigma:.3f} kappa={th.kappa:.2f} "
        f"sisters=({th.d0:.2f},{th.dinf:.2f},{th.tau_s:.1f}) K_j={th.Kj} fb={th.fb} rounding from {th.round_src}")
    if args.calib == "auto":
        CALIB = calibrate(args, th, args.workers)
    else:
        v = [float(x) for x in args.calib.split(",")]
        CALIB = dict(c=v[0], p_adj=v[1], hazard_gain=v[2] if len(v) > 2 else 1.0, speed_gain=v[3] if len(v) > 3 else 1.0,
                     source="manual")
    if args.speed_gain != "auto":
        CALIB["speed_gain"] = float(args.speed_gain)
    elif args.calib != "auto" and len(args.calib.split(",")) < 4:
        log("  NOTE: manual --calib without a 4th value -> speed_gain 1.0 (v3.7 motion); use --calib auto or pass the "
            "calibrated gain (PSC 0.92, Fluo 1.06 for theta/lib of 2026-10-10)")
    out_dir = args.out.rstrip("/")
    ctc_dir = out_dir + "_ctc"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ctc_dir, exist_ok=True)
    t0 = time.time()
    jobs = [(i, out_dir, ctc_dir) for i in range(args.num_clips)]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(min(args.workers, len(jobs))) as pl:
            metas = pl.map(_job, jobs)
    else:
        metas = [_job(j) for j in jobs]
    wall = time.time() - t0
    for m in metas:
        log(f"  {m['clip']}: cells/frame {m['mean_cells_per_frame']:.1f} div {m['n_div']} {m['division_frames']} "
            f"entries {m['n_entries']} exits {m['n_exits']} touch {m['touching_fraction']:.3f} "
            f"one_cc_viol {m['violations_one_cc']} lineage_viol {m['lineage_violations']} lock_viol {m['phase_lock_violations']} "
            f"merged {m['merged_single_child']} forced {m['forced_presence']} {m['seconds']:.1f}s")
    cf = sum(sum(m["cells_per_frame"][:-1]) for m in metas)
    summ = dict(dataset=args.dataset, generator=f"lineage_sim_v3.py {VERSION}", out=out_dir, ctc=ctc_dir,
                num_clips=len(metas), params=dict(vars(args)), calibration=CALIB, theta_sha=th.sha,
                mean_cells_per_frame=float(np.mean([m["mean_cells_per_frame"] for m in metas])),
                divisions_total=int(sum(m["n_div"] for m in metas)),
                divisions_per_clip_mean=float(np.mean([m["n_div"] for m in metas])),
                div_per_1000_atrisk=1000.0 * sum(m["n_div"] for m in metas) / max(cf, 1),
                theta_div_per_1000=1000 * th.rate,
                entry_per_1000_atrisk=1000.0 * sum(m["n_entries"] for m in metas) / max(cf, 1),
                exit_per_1000_atrisk=1000.0 * sum(m["n_exits"] for m in metas) / max(cf, 1),
                theta_entry_exit=[th.entry, th.exit],
                touching_fraction=float(np.mean([m["touching_fraction"] for m in metas])), theta_touching=th.touch,
                nn_median_px=float(np.median([m["nn_px_q10_50_90"][1] for m in metas])), theta_nn_q=list(th.nn_q),
                mean_contact_pairs_per_frame=float(np.mean([m["mean_contact_pairs_per_frame"] for m in metas])),
                overlap_violations=0,
                one_cc_violations=int(sum(m["violations_one_cc"] for m in metas)),
                one_cc_violations_8conn=int(sum(m["violations_one_cc_8conn"] for m in metas)),
                lineage_violations=int(sum(m["lineage_violations"] for m in metas)),
                phase_lock_violations=int(sum(m["phase_lock_violations"] for m in metas)),
                merged_single_child=int(sum(m["merged_single_child"] for m in metas)),
                dropped_links=int(sum(m["dropped_links"] for m in metas)),
                forced_presence=int(sum(m["forced_presence"] for m in metas)),
                cap_restored=int(sum(m["cap_restored"] for m in metas)),
                visible_area_loss_max=float(max(m["visible_area_loss_frac"]["max"] for m in metas)),
                occluded_px_total=int(sum(m["occluded_px"] for m in metas)),
                seconds_per_clip_single=float(np.mean([m["seconds"] for m in metas])), wall_seconds=wall)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    txt = "\n".join(f"{k}: {v}" for k, v in summ.items())
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    log(txt)
    if args.fig:
        montage(out_dir, metas, args.fig, args.num_frames)
        event_strip(out_dir, metas, args.fig.replace(".png", "_events.png"))
        log(f"  montage -> {args.fig} (+ _events.png)")


if __name__ == "__main__":
    main()
