#!/usr/bin/env python
"""verify_clips.py -- independent verifier for $N/data/vace_clips/<dataset> (and sim clips).

Recomputes from the raw CTC data (video 02 tif + ST masks + man_track.txt) what every clip file must contain
and compares byte-exactly. Independent re-implementation (does not import make_clips).
Usage: python verify_clips.py --dataset PhC-C2DL-PSC [--clips N (0=all)] [--root $N/data/vace_clips]
"""
import argparse, csv, glob, json, os, sys
import numpy as np, tifffile, cv2
from scipy import ndimage as ndi
from multiprocessing import Pool

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
CTC = f"{N}/data/ctc"
SRC_HW = {"PhC-C2DL-PSC": (576, 720), "Fluo-N2DL-HeLa": (700, 1100), "DIC-C2DH-HeLa": (512, 512), "Lead-BF": (2048, 2448)}


def nearest_rank_percentile(sorted_hist_cdf, n, p):
    k = max(1, int(np.ceil(p / 100.0 * n)))
    return int(np.searchsorted(sorted_hist_cdf, k))


def inner_boundary_4(lab):
    """pixel is on inner boundary if any 4-neighbour inside the image has a different label (image edge is not a boundary)."""
    fg = lab > 0
    # skimage find_boundaries(mode='inner') convention: the image border is NOT a boundary (reflect padding)
    pad = np.pad(lab, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    diff = (pad[:-2, 1:-1] != c) | (pad[2:, 1:-1] != c) | (pad[1:-1, :-2] != c) | (pad[1:-1, 2:] != c)
    return diff & fg


def check_clip(args):
    ds, cdir, tracks, mothers, daughters, lo, hi = args
    errs = []
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    s = meta["start_frame"]; nf = meta["num_frames"]; crop = meta["crop"]
    img_dir = f"{CTC}/{ds}/02"; seg_dir = f"{CTC}/{ds}/02_ST/SEG"
    present = {}
    ncc8_bad = []
    for tl in range(nf):
        tg = s + tl
        im = tifffile.imread(f"{img_dir}/t{tg:03d}.tif")
        lab = tifffile.imread(f"{seg_dir}/man_seg{tg:03d}.tif").astype(np.int64)
        if crop:
            y0, x0, h, w = crop["y0"], crop["x0"], crop["h"], crop["w"]
            im = im[y0:y0 + h, x0:x0 + w]; lab = lab[y0:y0 + h, x0:x0 + w]
        # --- files are real PNGs
        for sub in ("target", "cond", "labels"):
            p = os.path.join(cdir, sub, f"frame_{tl:03d}.png")
            with open(p, "rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    errs.append(f"{p}: not PNG magic")
        tgt = cv2.imread(os.path.join(cdir, "target", f"frame_{tl:03d}.png"), cv2.IMREAD_UNCHANGED)
        cond = cv2.imread(os.path.join(cdir, "cond", f"frame_{tl:03d}.png"), cv2.IMREAD_UNCHANGED)
        labp = cv2.imread(os.path.join(cdir, "labels", f"frame_{tl:03d}.png"), cv2.IMREAD_UNCHANGED)
        if tgt is None or cond is None or labp is None:
            errs.append(f"{cdir} f{tl}: unreadable png"); continue
        if tgt.shape != (lab.shape[0], lab.shape[1], 3) or tgt.dtype != np.uint8:
            errs.append(f"{cdir} f{tl}: target shape/dtype {tgt.shape} {tgt.dtype} expected {lab.shape}+3 uint8")
        if cond.shape != (lab.shape[0], lab.shape[1], 3) or cond.dtype != np.uint8:
            errs.append(f"{cdir} f{tl}: cond shape/dtype {cond.shape} {cond.dtype}")
        if labp.shape != lab.shape or labp.dtype != np.uint16:
            errs.append(f"{cdir} f{tl}: labels shape/dtype {labp.shape} {labp.dtype}")
        # --- labels == cropped ST
        if not np.array_equal(labp.astype(np.int64), lab):
            errs.append(f"{cdir} f{tl}: labels png != ST mask")
        # --- target == fixed mapping, gray replicated
        u8 = np.round(np.clip((im.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        if not (np.array_equal(tgt[..., 0], u8) and np.array_equal(tgt[..., 1], u8) and np.array_equal(tgt[..., 2], u8)):
            d = np.abs(tgt[..., 0].astype(int) - u8.astype(int))
            errs.append(f"{cdir} f{tl}: target != mapping(raw) maxdiff={d.max()} n={int((d>0).sum())}")
        if tl == 0:
            ref = cv2.imread(os.path.join(cdir, "ref.png"), cv2.IMREAD_UNCHANGED)
            if ref is None or not np.array_equal(ref, tgt):
                errs.append(f"{cdir}: ref.png != target frame 0")
        # --- cond channels (cv2 gives BGR)
        R, G, B = cond[..., 2], cond[..., 1], cond[..., 0]
        bnd = inner_boundary_4(lab)
        if not np.array_equal(R > 0, bnd) or not np.all(np.isin(R, [0, 255])):
            errs.append(f"{cdir} f{tl}: R channel != 4-conn inner boundary (diff {int((( R>0)!=bnd).sum())})")
        Gexp = np.zeros(lab.shape, np.uint8); Bexp = np.zeros(lab.shape, np.uint8)
        ids = np.unique(lab); ids = ids[ids > 0]
        for lid in ids:
            lid = int(lid)
            m = lab == lid
            present.setdefault(lid, []).append(tl)
            edt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
            mx = edt.max()
            Gexp[m] = np.round(edt[m] / mx * 255).astype(np.uint8)
            if lid in mothers and mothers[lid] - 2 <= tg <= mothers[lid]:
                Bexp[m] = 255
            elif lid in daughters and daughters[lid] == tg:
                Bexp[m] = 128
            if ndi.label(m, structure=np.ones((3, 3)))[1] != 1:
                ncc8_bad.append((tl, lid))
        if not np.array_equal(G, Gexp):
            d = np.abs(G.astype(int) - Gexp.astype(int))
            errs.append(f"{cdir} f{tl}: G channel != per-instance EDT (maxdiff {d.max()}, n {int((d>0).sum())})")
        if not np.array_equal(B, Bexp):
            errs.append(f"{cdir} f{tl}: B channel != event markers (n255 got {int((B==255).sum())} exp {int((Bexp==255).sum())}; n128 got {int((B==128).sum())} exp {int((Bexp==128).sum())})")
        if np.any((G > 0) & (lab == 0)) or np.any((G == 0) & (lab > 0)):
            errs.append(f"{cdir} f{tl}: G support != foreground")
    # --- lineage.txt
    lin = np.loadtxt(os.path.join(cdir, "lineage.txt"), dtype=np.int64, ndmin=2)
    lin_d = {int(L): (int(Bq), int(E), int(P)) for L, Bq, E, P in lin}
    if set(lin_d) != set(present):
        errs.append(f"{cdir}: lineage ids != label ids (missing {set(present)-set(lin_d)}, extra {set(lin_d)-set(present)})")
    for L, (Bq, E, P) in lin_d.items():
        if L in present and (Bq != present[L][0] or E != present[L][-1]):
            errs.append(f"{cdir}: id {L} B/E {Bq},{E} != presence {present[L][0]},{present[L][-1]}")
        gP = tracks.get(L, (0, 0, 0))[2]
        expP = gP if (gP > 0 and gP in present) else 0
        if P != expP:
            errs.append(f"{cdir}: id {L} parent {P} != expected {expP}")
        if P > 0:
            if lin_d[P][1] >= Bq:
                errs.append(f"{cdir}: parent {P} ends at {lin_d[P][1]} >= child {L} birth {Bq}")
        if L not in tracks or not (tracks[L][0] <= s + Bq and s + E <= tracks[L][1]):
            errs.append(f"{cdir}: id {L} outside global lifetime")
    # --- mp4 decode
    try:
        import imageio.v2 as iio
        for name, folder in (("target.mp4", "target"), ("cond.mp4", "cond")):
            rd = iio.get_reader(os.path.join(cdir, name))
            fr = [f for f in rd]
            rd.close()
            if len(fr) != nf:
                errs.append(f"{cdir}/{name}: {len(fr)} frames != {nf}")
            maes = []
            for tl in (0, nf // 2, nf - 1):
                png = cv2.imread(os.path.join(cdir, folder, f"frame_{tl:03d}.png"))[..., ::-1]
                if fr[tl].shape != png.shape:
                    errs.append(f"{cdir}/{name}: frame shape {fr[tl].shape} != {png.shape}")
                maes.append(float(np.abs(fr[tl].astype(float) - png.astype(float)).mean()))
            meta[f"_mae_{folder}"] = maes
    except Exception as e:
        errs.append(f"{cdir}: mp4 decode error {e}")
    gaps = [L for L, fr in present.items() if fr[-1] - fr[0] + 1 != len(fr)]
    # division count (my own def): mother present in her last frame E, E in [s, s+nf-2], >=2 daughters present at E+1
    ndiv = 0
    for mid, E in mothers.items():
        if mid in present and (E - s) in present[mid] and s <= E < s + nf - 1:
            nd = sum(1 for d, b in daughters.items() if tracks[d][2] == mid and b == E + 1 and d in present and (b - s) in present[d])
            ndiv += nd >= 2
    stats = dict(clip=os.path.basename(cdir), n_ids=len(present), div=ndiv, meta_div=meta["divisions_in_clip"],
                 gaps=len(gaps), meta_gaps=meta["ids_with_temporal_gaps"], cc8_bad=len(ncc8_bad), meta_cc=meta["violations_one_cc"],
                 cc8_list=ncc8_bad, mae_target=meta.get("_mae_target"), mae_cond=meta.get("_mae_cond"), start=s)
    if ndiv != meta["divisions_in_clip"]: errs.append(f"{cdir}: my div {ndiv} != meta {meta['divisions_in_clip']}")
    if len(gaps) != meta["ids_with_temporal_gaps"]: errs.append(f"{cdir}: my gaps {len(gaps)} != meta")
    if len(ncc8_bad) != meta["violations_one_cc"]: errs.append(f"{cdir}: my cc {len(ncc8_bad)} != meta")
    return stats, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=f"{N}/data/vace_clips")
    ap.add_argument("--clips", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ds = a.dataset; root = os.path.join(a.root, ds)
    mp = json.load(open(os.path.join(root, "intensity_mapping.json")))
    # --- independent percentile recomputation
    files = sorted(glob.glob(f"{CTC}/{ds}/02/t*.tif"))
    allpix = np.concatenate([tifffile.imread(f).ravel() for f in files])
    n = allpix.size
    srt = np.sort(allpix)
    lo = int(srt[max(1, int(np.ceil(0.01 * n))) - 1]); hi = int(srt[max(1, int(np.ceil(0.999 * n))) - 1])
    print(f"[{ds}] my nearest-rank percentiles over {n} px: lo={lo} hi={hi} | json lo={mp['lo']} hi={mp['hi']} raw[{srt[0]},{srt[-1]}] json raw[{mp['raw_min']},{mp['raw_max']}]")
    print(f"   clipped below {float((allpix<lo).mean()):.5f} above {float((allpix>hi).mean()):.5f} | json {mp['frac_pixels_clipped_below_lo']:.5f} {mp['frac_pixels_clipped_above_hi']:.5f}")
    assert lo == mp["lo"] and hi == mp["hi"], "PERCENTILE MISMATCH"
    del allpix, srt
    tr = np.loadtxt(f"{CTC}/{ds}/02_GT/TRA/man_track.txt", dtype=np.int64, ndmin=2)
    tracks = {int(L): (int(B), int(E), int(P)) for L, B, E, P in tr}
    ch = {}
    for L, (B, E, P) in tracks.items():
        if P > 0: ch.setdefault(P, []).append(L)
    mothers = {P: tracks[P][1] for P, c in ch.items() if len(c) >= 2 and P in tracks}
    daughters = {d: tracks[d][0] for P in mothers for d in ch[P]}
    single = sum(1 for P, c in ch.items() if len(c) == 1)
    T = len(files)
    print(f"   tracks={len(tracks)} divisions={len(mothers)} single-child links={single} frames={T}")
    # metadata.csv
    rows = list(csv.DictReader(open(os.path.join(root, "metadata.csv"))))
    clips = sorted(d for d in os.listdir(root) if d.startswith("clip_"))
    print(f"   metadata rows={len(rows)} cols={list(rows[0].keys())} clip dirs={len(clips)}")
    bad = [r for r in rows if not all(os.path.exists(os.path.join(root, r[k])) for k in ("video", "vace_video", "vace_reference_image"))]
    print(f"   metadata rows with missing files: {len(bad)}; first row: {rows[0]}")
    # coverage of divisions by clip windows (mother last frame E with E in [s, s+nf-2])
    nf = 17; stride = 4
    starts = [int(json.load(open(os.path.join(root, c, 'meta.json')))['start_frame']) for c in clips]
    starts_u = sorted(set(starts))
    cov = sum(1 for E in mothers.values() if any(s <= E < s + nf - 1 for s in starts_u))
    print(f"   unique starts={len(starts_u)} ({starts_u[0]}..{starts_u[-1]}), divisions whose mother-last-frame is inside a clip window: {cov}/{len(mothers)}; uncovered E: {sorted(E for E in mothers.values() if not any(s <= E < s + nf - 1 for s in starts_u))}")
    if a.clips: clips = clips[: a.clips]
    jobs = [(ds, os.path.join(root, c), tracks, mothers, daughters, lo, hi) for c in clips]
    with Pool(a.workers) as pool:
        res = pool.map(check_clip, jobs, chunksize=1)
    errs = [e for _, es in res for e in es]
    stats = [s for s, _ in res]
    print(f"   checked {len(stats)} clips; ERRORS: {len(errs)}")
    for e in errs[:40]: print("     ERR", e)
    print(f"   sum div={sum(s['div'] for s in stats)} clips_with_div={sum(s['div']>0 for s in stats)} min/max div={min(s['div'] for s in stats)}/{max(s['div'] for s in stats)}")
    print(f"   sum gaps={sum(s['gaps'] for s in stats)} sum cc8_bad={sum(s['cc8_bad'] for s in stats)} cc8 pairs (global frame,id): {sorted(set((s['start']+t, l) for s in stats for t, l in s['cc8_list']))}")
    mt = [m for s in stats if s['mae_target'] for m in s['mae_target']]; mc = [m for s in stats if s['mae_cond'] for m in s['mae_cond']]
    if mt: print(f"   mp4 MAE target: min {min(mt):.2f} max {max(mt):.2f} mean {np.mean(mt):.2f}; cond: min {min(mc):.2f} max {max(mc):.2f} mean {np.mean(mc):.2f}")
    print("   VERDICT:", "OK" if not errs else "FAIL")


if __name__ == "__main__":
    main()
