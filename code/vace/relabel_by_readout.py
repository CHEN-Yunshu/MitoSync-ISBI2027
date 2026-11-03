#!/usr/bin/env python
"""relabel_by_readout.py -- make a synthetic clip's lineage labels consistent with the generated images.

Run with the cellres_cp4_blackwell env (cellpose 4.0.4).  Never modifies the input corpus; writes a NEW twin set.

    python relabel_by_readout.py --ctc_root <tag>_ctc --out_root <tag>_ctc_rl --gpu 0 --workers 4

Per clip
  1. Cellpose-SAM (cpsam, diameter None, flow 0.4, cellprob 0, min_size 15, normalize) on <clip>/t%03d.tif,
     cached as <out_root>/<clip>/cpsam.npz {"masks": uint16 (T,H,W)} (reruns skip segmentation).
  2. Predictions are matched to the ST labels per frame (IoU >= 0.5, one-to-one, greedy by IoU; same rule as
     gen_eval.match_frame / readout_f1).
  3. For every division (parent with exactly 2 daughters born at the same tb = parent E + 1):
       t* = first frame >= tb where BOTH daughters are matched in t* AND in t*+1 (searched to the clip end).
       t* == tb : division kept at birth (nothing changes).
       t* >  tb : the mother is extended to t*-1: pixels of both daughters in frames tb..t*-1 become the mother id,
                  mother E = t*-1, daughters B = t*  ("delayed" by t*-tb frames).
       no t*    : the daughters are merged into the mother for the whole clip (mother E = max daughter E, daughters
                  removed, their pixels relabelled as mother) -> "dropped".  Children of a dropped daughter are
                  re-parented to the mother; if they form a division it is dropped as well ("dropped_cascade",
                  only possible in real clips: the simulator never divides a daughter inside a clip).
     A merged mask that is not 4-connected (Fluo daughters are born non-touching) is bridged along the shortest
     4-connected path through BACKGROUND pixels (geodesic; other cells are impassable), widened to 3 px where the
     background allows.  If a fragment is walled in by other cells, the cheapest path may cross at most 8 pixels of
     neighbouring cells (only if they stay one component); such events are counted (bridge_through_cells_frames,
     bridge_foreign_pixels) in the summary.  If no admissible path exists either, the smaller fragment is set to
     background (bridge_fragment_to_bg_frames / _pixels, fragment_events per clip) so the id stays one component.
  4. Non-division tracks are unchanged.
  5. Writes <out_root>/<clip>_ST/SEG/man_seg%03d.tif, <clip>_GT/TRA/man_track%03d.tif (disc markers, same
     markers_image() as lineage_sim.py / real_clip_ctc.py; marker-only gap frames of real clips keep their original
     marker, counted as markers_carried) + man_track.txt, and symlinks (or --copy_images) the image tifs into
     <out_root>/<clip>/.
  6. ctc_io.verify_ctc on every output clip; <out_root>/relabel_summary.json (per clip + aggregate) and
     <out_root>/accepted_dirs.txt (readout F1 against the NEW labels >= --f1_thr, default 0.75 for PhC-C2DL-PSC and
     no threshold for Fluo-N2DL-HeLa, and verify_ctc clean).
"""
import os, sys, json, glob, time, argparse, shutil, traceback
from collections import defaultdict, Counter

import numpy as np
import tifffile
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ctc_io import verify_ctc, read_tracks, write_tracks  # noqa: E402

S8 = np.ones((3, 3), bool)
S4 = ndi.generate_binary_structure(2, 1)  # verify_ctc / Trackastra-style loader use 4-connectivity
MARKER_R = {"PhC-C2DL-PSC": 3, "Fluo-N2DL-HeLa": 4, "DIC-C2DH-HeLa": 10, "Lead-BF": 7}
CP_SETTINGS = dict(diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, normalize=True)


def log(*a):
    print("[relabel]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- markers (verbatim)
def markers_image(lab, radius):
    """CTC TRA markers: disc of `radius` at each instance centroid, restricted to the instance mask (lineage_sim.py)."""
    out = np.zeros_like(lab)
    rr = np.arange(-radius, radius + 1)
    disc = (rr[:, None] ** 2 + rr[None, :] ** 2) <= radius ** 2
    H, W = lab.shape
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        lid = i + 1
        sub = lab[sl] == lid
        if not sub.any():
            continue
        ys, xs = np.nonzero(sub)
        cy, cx = ys.mean(), xs.mean()
        if not sub[int(round(cy)), int(round(cx))]:  # centroid outside (concave): nearest mask pixel
            k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            cy, cx = ys[k], xs[k]
        cy, cx = int(round(cy)) + sl[0].start, int(round(cx)) + sl[1].start
        y0, x0 = max(cy - radius, 0), max(cx - radius, 0)
        y1, x1 = min(cy + radius + 1, H), min(cx + radius + 1, W)
        d = disc[y0 - (cy - radius):y1 - (cy - radius), x0 - (cx - radius):x1 - (cx - radius)]
        win = out[y0:y1, x0:x1]
        m = d & (lab[y0:y1, x0:x1] == lid)
        cc, k = ndi.label(m, structure=S8)
        if k > 1:
            m = cc == cc[cy - y0, cx - x0]
        win[m] = lid
        if not (out == lid).any():
            out[cy, cx] = lid
    return out


# ----------------------------------------------------------------------------------------------- matching
def match_frame(lab, pred, thr=0.5):
    """Greedy one-to-one IoU matching (gen_eval.match_frame). Returns (dict lab_id->pred_id, n_lab, n_pred)."""
    L = lab.ravel().astype(np.int64); P = pred.ravel().astype(np.int64)
    area_l = np.bincount(L); area_p = np.bincount(P)
    n_lab = int((area_l[1:] > 0).sum()); n_pred = int((area_p[1:] > 0).sum())
    both = (L > 0) & (P > 0)
    if not both.any():
        return {}, n_lab, n_pred
    base = int(P.max()) + 1
    pairs, cnt = np.unique(L[both] * base + P[both], return_counts=True)
    li, pi = pairs // base, pairs % base
    iou = cnt / (area_l[li] + area_p[pi] - cnt)
    order = np.argsort(-iou, kind="stable")
    used_l, used_p, m = set(), set(), {}
    for k in order:
        if iou[k] < thr:
            break
        a, b = int(li[k]), int(pi[k])
        if a in used_l or b in used_p:
            continue
        used_l.add(a); used_p.add(b); m[a] = b
    return m, n_lab, n_pred


def f1_of(m, n_lab, n_pred):
    tp = len(m); fp = n_pred - tp; fn = n_lab - tp
    if n_lab == 0 and n_pred == 0:
        return 1.0
    return 2 * tp / max(2 * tp + fp + fn, 1)


# ----------------------------------------------------------------------------------------------- bridging
def ensure_connected(lab, lid, max_iter=8, max_foreign=8):
    """Make `lab == lid` one 4-connected component (verify_ctc connectivity).
    Each iteration joins the largest component to the component nearest to it along the shortest 4-connected path
    through BACKGROUND pixels (geodesic, other cells impassable), widened to 3 px where the background allows.
    Last resort (component walled in by other cells): the cheapest path may cross at most `max_foreign` pixels of
    neighbouring cells, and only if every touched neighbour stays one 4-connected component (else reverted).
    Final fallback (no admissible path): the smaller fragment(s) are relabelled to BACKGROUND (counted as
    fragment_to_bg pixels) so that the id stays one component.
    Returns (status, n_added_background, n_foreign_pixels, n_fragment_to_bg): status 0 = already connected,
    1 = bridged, 2 = fragment(s) dropped to background, -1 = failed."""
    m = lab == lid
    cc, k = ndi.label(m, structure=S4)
    if k <= 1:
        return 0, 0, 0, 0
    from skimage.graph import route_through_array
    added = foreign = 0
    for _ in range(max_iter):
        cc, k = ndi.label(lab == lid, structure=S4)
        if k <= 1:
            return 1, added, foreign, 0
        sizes = np.bincount(cc.ravel())[1:]
        main = int(np.argmax(sizes)) + 1
        A = cc == main
        dist, (iy, ix) = ndi.distance_transform_edt(~A, return_indices=True)
        others = (cc > 0) & ~A
        oy, ox = np.nonzero(others)
        j = int(np.argmin(dist[oy, ox]))
        p1 = (int(oy[j]), int(ox[j])); p0 = (int(iy[p1]), int(ix[p1]))
        free = (lab == 0) | (lab == lid)
        y0, y1 = min(p0[0], p1[0]), max(p0[0], p1[0]); x0, x1 = min(p0[1], p1[1]), max(p0[1], p1[1])
        path = None
        for pad in (16, 64, 100000):  # window around the endpoints (speed); widened if no background path is found
            ya, yb = max(y0 - pad, 0), min(y1 + pad + 1, lab.shape[0]); xa, xb = max(x0 - pad, 0), min(x1 + pad + 1, lab.shape[1])
            cost = np.where(free[ya:yb, xa:xb], 1.0, 1e6)
            idx, c = route_through_array(cost, (p0[0] - ya, p0[1] - xa), (p1[0] - ya, p1[1] - xa), fully_connected=False, geometric=False)
            if c < 1e6:
                path = np.array(idx) + [ya, xa]; break
        if path is not None:
            seg = np.zeros(lab.shape, bool); seg[path[:, 0], path[:, 1]] = True
            seg = ndi.binary_dilation(seg, structure=S8, iterations=1) & (lab == 0)
            if not seg.any():
                break
            lab[seg] = lid; added += int(seg.sum()); continue
        # last resort: cross as few foreign pixels as possible
        pad = 64
        ya, yb = max(y0 - pad, 0), min(y1 + pad + 1, lab.shape[0]); xa, xb = max(x0 - pad, 0), min(x1 + pad + 1, lab.shape[1])
        cost = np.where(free[ya:yb, xa:xb], 1.0, 50.0)
        idx, c = route_through_array(cost, (p0[0] - ya, p0[1] - xa), (p1[0] - ya, p1[1] - xa), fully_connected=False, geometric=False)
        path = np.array(idx) + [ya, xa]
        vals = lab[path[:, 0], path[:, 1]]
        fpix = path[(vals != 0) & (vals != lid)]
        if len(fpix) == 0 or len(fpix) > max_foreign:
            break
        touched = set(int(v) for v in vals if v not in (0, lid))
        backup = lab[path[:, 0], path[:, 1]].copy()
        lab[path[:, 0], path[:, 1]] = lid
        bg = np.zeros(lab.shape, bool); bg[path[:, 0], path[:, 1]] = True
        bg = ndi.binary_dilation(bg, structure=S8, iterations=1) & (lab == 0)
        lab[bg] = lid
        ok = all((lab == x).any() and ndi.label(lab == x, structure=S4)[1] == 1 for x in touched)
        if not ok:
            lab[bg] = 0; lab[path[:, 0], path[:, 1]] = backup
            break
        added += int(bg.sum()) + int((backup == 0).sum()); foreign += int(len(fpix))  # path background pixels were set to lid above (verifier fix: were uncounted)
    cc, k = ndi.label(lab == lid, structure=S4)
    if k <= 1:
        return 1, added, foreign, 0
    sizes = np.bincount(cc.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    frag = (cc > 0) & (cc != main)
    lab[frag] = 0
    k = ndi.label(lab == lid, structure=S4)[1]
    return (2 if k == 1 else -1), added, foreign, int(frag.sum())


# ----------------------------------------------------------------------------------------------- per-clip relabel
def relabel_clip(clip_dir, out_root, rad, iou_thr, copy_images):
    t0 = time.time()
    clip_dir = os.path.abspath(clip_dir); name = os.path.basename(clip_dir); parent = os.path.dirname(clip_dir)
    gt = os.path.join(parent, f"{name}_GT", "TRA"); st = os.path.join(parent, f"{name}_ST", "SEG")
    img_files = sorted(glob.glob(os.path.join(clip_dir, "t*.tif")))
    T = len(img_files)
    seg = np.stack([tifffile.imread(os.path.join(st, f"man_seg{t:03d}.tif")).astype(np.int32) for t in range(T)])
    pred = np.load(os.path.join(out_root, name, "cpsam.npz"))["masks"].astype(np.int32)
    assert pred.shape == seg.shape, (pred.shape, seg.shape)
    tracks = read_tracks(os.path.join(gt, "man_track.txt"))
    byid = {int(L): [int(B), int(E), int(P)] for L, B, E, P in tracks}

    # 2. matching of predictions to ORIGINAL labels
    matched, f1_orig = [], []
    for t in range(T):
        m, nl, npd = match_frame(seg[t], pred[t], iou_thr)
        matched.append(set(m.keys())); f1_orig.append(f1_of(m, nl, npd))

    # 3. divisions
    children = defaultdict(list)
    for L, (B, E, P) in byid.items():
        if P:
            children[P].append(L)
    divs = []
    for P, ch in children.items():
        if len(ch) == 2:
            divs.append((min(byid[c][0] for c in ch), P, sorted(ch)))
    divs.sort()
    new = seg.copy(); nb = {L: list(v) for L, v in byid.items()}
    alias = {}  # removed id -> surviving mother id
    st_ = Counter(); delays = []; records = []
    changed = np.zeros(T, bool); bridge_added = 0; frag_events = []

    def merge_into(frame_ids, src_ids, dst):
        """relabel src_ids -> dst in the given frames, keep dst one component. returns n pixels relabelled."""
        nonlocal bridge_added
        npx = 0
        for t in frame_ids:
            mk = np.isin(new[t], src_ids)
            n = int(mk.sum())
            if n == 0:
                continue
            new[t][mk] = dst; npx += n; changed[t] = True
            status, added, foreign, frag = ensure_connected(new[t], dst)
            if status in (1, 2):
                st_["bridged_frames"] += 1; bridge_added += added
                if foreign:
                    st_["bridge_through_cells_frames"] += 1; st_["bridge_foreign_pixels"] += foreign
                if status == 2:
                    st_["bridge_fragment_to_bg_frames"] += 1; st_["bridge_fragment_to_bg_pixels"] += frag
                    frag_events.append(dict(t=t, id=int(dst), pixels=frag))
            elif status == -1:
                st_["bridge_failed_frames"] += 1
        return npx

    for tb, P, (d1, d2) in divs:
        st_["divisions_original"] += 1
        rec = dict(parent=P, daughters=[d1, d2], tb=tb)
        if byid[d1][0] != byid[d2][0] or byid[P][1] != tb - 1:
            st_["skipped_irregular"] += 1; rec["outcome"] = "skipped_irregular"; records.append(rec); continue
        if P in alias:  # parent was merged away by a dropped division -> cascade drop into the survivor
            M = alias[P]
            frames = range(tb, max(byid[d1][1], byid[d2][1]) + 1)
            npx = merge_into(frames, [d1, d2], M)
            nb[M][1] = max(nb[M][1], byid[d1][1], byid[d2][1])
            for d in (d1, d2):
                alias[d] = M; nb.pop(d, None)
            st_["dropped_cascade"] += 1; st_["pixels_relabelled"] += npx
            rec.update(outcome="dropped_cascade", into=M, pixels=npx); records.append(rec); continue
        t_star = None
        for t in range(tb, T - 1):
            if d1 in matched[t] and d2 in matched[t] and d1 in matched[t + 1] and d2 in matched[t + 1]:
                t_star = t; break
        if t_star == tb:
            st_["kept_at_birth"] += 1; rec.update(outcome="kept", t_star=t_star)
        elif t_star is not None:
            npx = merge_into(range(tb, t_star), [d1, d2], P)
            nb[P][1] = t_star - 1; nb[d1][0] = t_star; nb[d2][0] = t_star
            delays.append(t_star - tb); st_["delayed"] += 1; st_["pixels_relabelled"] += npx
            rec.update(outcome="delayed", t_star=t_star, delay=t_star - tb, pixels=npx)
        else:
            frames = range(tb, max(byid[d1][1], byid[d2][1]) + 1)
            npx = merge_into(frames, [d1, d2], P)
            nb[P][1] = max(nb[P][1], byid[d1][1], byid[d2][1])
            for d in (d1, d2):
                alias[d] = P; nb.pop(d, None)
            st_["dropped"] += 1; st_["pixels_relabelled"] += npx
            rec.update(outcome="dropped", pixels=npx)
        records.append(rec)

    # re-parent children of removed ids (non-division children / 3+ children / single links); break impossible links
    for L in list(nb):
        Pp = nb[L][2]
        if Pp in alias:
            M = alias[Pp]
            while M in alias:
                M = alias[M]
            if M in nb and nb[M][1] < nb[L][0]:
                nb[L][2] = M; st_["links_reparented"] += 1
            else:
                nb[L][2] = 0; st_["links_broken"] += 1
    # markers: regenerated from the new masks; real clips may have gap frames where a track exists only as a TRA
    # marker (no ST mask) -> carry the ORIGINAL marker over (mapped to the surviving id) so B..E stays complete.
    surv = {}
    for a, m in alias.items():
        while m in alias:
            m = alias[m]
        surv.setdefault(m, []).append(a)
    markers = []
    for t in range(T):
        L = new[t].astype(np.uint16); M = markers_image(L, rad)
        exp = {i for i, (B, E, P) in nb.items() if B <= t <= E}
        missing = exp - set(int(i) for i in np.unique(L) if i)
        if missing:
            tra_o = tifffile.imread(os.path.join(gt, f"man_track{t:03d}.tif")).astype(np.int32)
            for lid in missing:
                mk = np.isin(tra_o, [lid] + surv.get(lid, [])) & (L == 0) & (M == 0)
                if mk.any():
                    cc, k = ndi.label(mk, structure=S4)
                    if k > 1:
                        mk = cc == (int(np.argmax(np.bincount(cc.ravel())[1:])) + 1)
                    M[mk] = lid; st_["markers_carried"] += 1
        markers.append(M)
    # sanity: every surviving id present exactly in B..E of max(markers, masks) (what verify_ctc / the loader see)
    for L, (B, E, P) in nb.items():
        pres = [t for t in range(T) if (new[t] == L).any() or (markers[t] == L).any()]
        if pres != list(range(B, E + 1)):
            st_["track_extent_mismatch"] += 1

    # 5. write outputs
    o_img = os.path.join(out_root, name); o_tra = os.path.join(out_root, f"{name}_GT", "TRA"); o_seg = os.path.join(out_root, f"{name}_ST", "SEG")
    for d in (o_tra, o_seg):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    os.makedirs(o_img, exist_ok=True)
    for f in img_files:
        dst = os.path.join(o_img, os.path.basename(f))
        if os.path.lexists(dst):
            os.remove(dst)
        if copy_images:
            shutil.copy(os.path.realpath(f), dst)
        else:
            os.symlink(os.path.realpath(f), dst)
    f1_new = []
    for t in range(T):
        L = new[t].astype(np.uint16)
        tifffile.imwrite(os.path.join(o_seg, f"man_seg{t:03d}.tif"), L, compression="zlib")
        tifffile.imwrite(os.path.join(o_tra, f"man_track{t:03d}.tif"), markers[t].astype(np.uint16), compression="zlib")
        m, nl, npd = match_frame(new[t], pred[t], iou_thr); f1_new.append(f1_of(m, nl, npd))
    write_tracks(os.path.join(o_tra, "man_track.txt"), [(L, B, E, P) for L, (B, E, P) in sorted(nb.items())])

    # 6. verify
    v = verify_ctc(o_img)
    v_ok = v["bad_frames"] == 0 and v["bad_parents"] == 0 and v["missing_ids"] == 0 and v["extra_ids"] == 0 and v["multi_component"] == 0
    fg_px = int((seg > 0).sum())
    return dict(clip=name, src=clip_dir, out=o_img, frames=T, tracks_original=len(byid), tracks_new=len(nb),
                divisions_original=st_["divisions_original"], kept_at_birth=st_["kept_at_birth"], delayed=st_["delayed"],
                delay_mean=float(np.mean(delays)) if delays else None, delays=delays, dropped=st_["dropped"],
                dropped_cascade=st_["dropped_cascade"], skipped_irregular=st_["skipped_irregular"],
                links_reparented=st_["links_reparented"], links_broken=st_["links_broken"], track_extent_mismatch=st_["track_extent_mismatch"],
                markers_carried=st_["markers_carried"],
                frames_relabelled=int(changed.sum()), pixels_relabelled=int(st_["pixels_relabelled"]), bridge_pixels_added=int(bridge_added),
                bridged_frames=st_["bridged_frames"], bridge_failed_frames=st_["bridge_failed_frames"],
                bridge_through_cells_frames=st_["bridge_through_cells_frames"], bridge_foreign_pixels=st_["bridge_foreign_pixels"],
                bridge_fragment_to_bg_frames=st_["bridge_fragment_to_bg_frames"], bridge_fragment_to_bg_pixels=st_["bridge_fragment_to_bg_pixels"],
                fragment_events=frag_events, foreground_pixels=fg_px,
                readout_f1_orig=float(np.mean(f1_orig)), readout_f1_new=float(np.mean(f1_new)),
                readout_f1_orig_per_frame=[round(x, 4) for x in f1_orig], readout_f1_new_per_frame=[round(x, 4) for x in f1_new],
                verify=v, verify_ok=bool(v_ok), divisions=records, seconds=round(time.time() - t0, 1))


def _worker(a):
    try:
        return relabel_clip(*a)
    except Exception as e:
        return dict(clip=os.path.basename(a[0]), src=a[0], error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())


# ----------------------------------------------------------------------------------------------- segmentation
def load_frames(clip_dir):
    files = sorted(glob.glob(os.path.join(clip_dir, "t*.tif")))
    out = []
    for f in files:
        a = tifffile.imread(f)
        if a.ndim == 3:
            a = a[..., :3].astype(np.float32).mean(-1)
        out.append(a.astype(np.float32))
    return out


def segment_clip(model, clip_dir, out_npz, batch):
    frames = load_frames(clip_dir)
    masks = model.eval(frames, batch_size=batch, channel_axis=None, **CP_SETTINGS)[0]
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    masks = np.stack([np.asarray(m) for m in masks]).astype(np.uint16)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    tmp = out_npz + ".tmp.npz"
    np.savez_compressed(tmp, masks=masks); os.replace(tmp, out_npz)
    return masks.shape, [int(len(np.unique(m)) - 1) for m in masks]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4, help="Cellpose batch size (GPUs are shared: keep small)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (sorted)")
    ap.add_argument("--clips", nargs="*", default=None, help="explicit clip names")
    ap.add_argument("--f1_thr", type=float, default=None, help="acceptance threshold on readout F1 vs NEW labels (default: 0.75 for PhC-C2DL-PSC, none otherwise)")
    ap.add_argument("--marker_radius", type=int, default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--copy_images", action="store_true")
    ap.add_argument("--seg_only", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ctc_root = os.path.abspath(args.ctc_root); out_root = os.path.abspath(args.out_root)
    assert os.path.realpath(out_root) != os.path.realpath(ctc_root), "out_root must differ from ctc_root"
    ds = next((k for k in MARKER_R if k in ctc_root), None)
    rad = args.marker_radius if args.marker_radius is not None else MARKER_R.get(ds)
    assert rad is not None, "cannot infer dataset from path: pass --marker_radius"
    f1_thr = args.f1_thr if args.f1_thr is not None else (0.75 if ds == "PhC-C2DL-PSC" else None)
    clips = sorted(d for d in os.listdir(ctc_root)
                   if os.path.isdir(os.path.join(ctc_root, d)) and not d.endswith(("_GT", "_ST"))
                   and os.path.exists(os.path.join(ctc_root, f"{d}_GT", "TRA", "man_track.txt"))
                   and os.path.exists(os.path.join(ctc_root, d, "t000.tif")))
    if args.clips:
        clips = [c for c in clips if c in set(args.clips)]
    if args.limit:
        clips = clips[:args.limit]
    os.makedirs(out_root, exist_ok=True)
    log(f"ctc_root={ctc_root} out_root={out_root} clips={len(clips)} dataset={ds} marker_r={rad} f1_thr={f1_thr} gpu={args.gpu} workers={args.workers}")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = None if args.seg_only else ctx.Pool(args.workers)
    pending = []

    todo = [c for c in clips if not os.path.exists(os.path.join(out_root, c, "cpsam.npz"))]
    model = None
    if todo:
        t0 = time.time()
        from cellpose import models
        model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
        log(f"Cellpose-SAM loaded in {time.time() - t0:.1f}s; {len(todo)}/{len(clips)} clips to segment")
    t_start = time.time()
    for i, c in enumerate(clips):
        npz = os.path.join(out_root, c, "cpsam.npz")
        if not os.path.exists(npz):
            t1 = time.time()
            shp, n = segment_clip(model, os.path.join(ctc_root, c), npz, args.batch)
            log(f"seg {i + 1}/{len(clips)} {c} {shp} cells/frame mean {np.mean(n):.1f} {time.time() - t1:.1f}s")
        if pool is not None:
            pending.append(pool.apply_async(_worker, ((os.path.join(ctc_root, c), out_root, rad, args.iou, args.copy_images),)))
    if model is not None:
        del model
    if args.seg_only:
        log("seg_only done"); return
    results = []
    for i, p in enumerate(pending):
        r = p.get()
        results.append(r)
        if "error" in r:
            log(f"ERROR {r['clip']}: {r['error']}\n{r['trace']}")
        else:
            log(f"rl {i + 1}/{len(pending)} {r['clip']} div {r['divisions_original']} kept {r['kept_at_birth']} delayed {r['delayed']} "
                f"(mean {r['delay_mean']}) dropped {r['dropped']}+{r['dropped_cascade']} px {r['pixels_relabelled']} bridged {r['bridged_frames']}/{r['bridge_failed_frames']} xcell {r['bridge_through_cells_frames']} frag2bg {r['bridge_fragment_to_bg_frames']} "
                f"F1 {r['readout_f1_orig']:.3f}->{r['readout_f1_new']:.3f} verify_ok {r['verify_ok']} {r['seconds']}s")
    pool.close(); pool.join()

    ok = [r for r in results if "error" not in r]
    delays = [d for r in ok for d in r["delays"]]
    agg = dict(clips=len(results), errors=len(results) - len(ok), frames=sum(r["frames"] for r in ok))
    for k in ("divisions_original", "kept_at_birth", "delayed", "dropped", "dropped_cascade", "skipped_irregular", "links_reparented",
              "links_broken", "track_extent_mismatch", "markers_carried", "frames_relabelled", "pixels_relabelled", "bridge_pixels_added", "bridged_frames",
              "bridge_failed_frames", "bridge_through_cells_frames", "bridge_foreign_pixels", "bridge_fragment_to_bg_frames",
              "bridge_fragment_to_bg_pixels", "foreground_pixels"):
        agg[k] = int(sum(r[k] for r in ok))
    nd = max(agg["divisions_original"], 1)
    agg.update(kept_frac=agg["kept_at_birth"] / nd, delayed_frac=agg["delayed"] / nd, dropped_frac=(agg["dropped"] + agg["dropped_cascade"]) / nd,
               delay_mean=float(np.mean(delays)) if delays else None, delay_hist={int(k): int(v) for k, v in sorted(Counter(delays).items())},
               frames_relabelled_frac=agg["frames_relabelled"] / max(agg["frames"], 1),
               pixels_relabelled_frac_of_foreground=agg["pixels_relabelled"] / max(agg["foreground_pixels"], 1),
               readout_f1_orig_mean=float(np.mean([r["readout_f1_orig"] for r in ok])) if ok else None,
               readout_f1_new_mean=float(np.mean([r["readout_f1_new"] for r in ok])) if ok else None,
               verify_ok_clips=int(sum(r["verify_ok"] for r in ok)),
               verify_bad_frames=int(sum(r["verify"]["bad_frames"] for r in ok)), verify_bad_parents=int(sum(r["verify"]["bad_parents"] for r in ok)),
               verify_multi_component=int(sum(r["verify"]["multi_component"] for r in ok)),
               verify_divisions_new=int(sum(r["verify"]["divisions"] for r in ok)))
    accepted = [r for r in ok if r["verify_ok"] and (f1_thr is None or r["readout_f1_new"] >= f1_thr)]
    agg.update(f1_thr=f1_thr, accepted=len(accepted),
               accepted_f1_orig_rule=int(sum(1 for r in ok if r["verify_ok"] and (f1_thr is None or r["readout_f1_orig"] >= f1_thr))))
    summary = dict(ctc_root=ctc_root, out_root=out_root, dataset=ds, marker_radius=rad, iou=args.iou, cellpose=dict(model="cpsam", **CP_SETTINGS),
                   aggregate=agg, clips={r["clip"]: r for r in results}, elapsed_s=round(time.time() - t_start, 1))
    with open(os.path.join(out_root, "relabel_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_root, "accepted_dirs.txt"), "w") as f:
        f.writelines(r["out"] + "\n" for r in accepted)
    log("AGGREGATE " + json.dumps(agg))
    log(f"wrote {out_root}/relabel_summary.json and accepted_dirs.txt ({len(accepted)} accepted) in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
