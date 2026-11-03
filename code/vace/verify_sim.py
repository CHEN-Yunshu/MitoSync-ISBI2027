"""Independent verifier for lineage_sim outputs (run with cellres python). Usage: verify_sim.py <sim_dir> [max_clips]"""
import sys, os, json, glob
import numpy as np, cv2, tifffile
from scipy import ndimage as ndi
sys.path.insert(0, "STUDY_ROOT/scripts/vace")
import make_clips as MC  # the clip-dataset builder's cond encoder (reference implementation)

N = os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace"))
sim = sys.argv[1].rstrip("/")
maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
ctc = sim + "_ctc"
CANVAS = {"PhC-C2DL-PSC": (576, 720), "Fluo-N2DL-HeLa": (512, 768), "DIC-C2DH-HeLa": (512, 512), "Lead-BF": (576, 720)}
# Lead-BF (added): comma-separated multi-video libraries (never 01), ref from any listed video, 3x3 grid of 576x720 ref
# crops, cond_v2 corpora (meta cond_dir / cond_version, checked against cond_render), optional intensity_mapping.json
import cond_render as CR
LEAD_CROPS = [(y0, x0, 576, 720) for y0 in (0, 736, 1472) for x0 in (0, 864, 1728)]
clips = sorted(d for d in os.listdir(sim) if d.startswith("clip_"))[:maxc]
errs = []
stats = dict(clips=0, frames=0, ids=0, divs=0, multi_cc4=0, multi_cc8=0, cond_mismatch_px=0, ref_mismatch=0,
             tra_outside=0, tra_missing=0, tra_multi=0, label_mismatch=0, lineage_bad=0, bad_png=0, jpg_files=0,
             bad_shape=0, mother_marked_frames=0, daughter_marked_frames=0, cells=[])
# any jpg anywhere?
for root, _, files in os.walk(os.path.dirname(sim)):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg")):
            stats["jpg_files"] += 1; errs.append("JPEG file: " + os.path.join(root, f))
for cid in clips:
    cdir = f"{sim}/{cid}"
    meta = json.load(open(f"{cdir}/meta.json"))
    ds = meta["dataset"]; H, W = CANVAS[ds]; nf = meta["num_frames"]
    lead = ds == "Lead-BF"
    if lead:
        if "01" in meta["video"].split(","): errs.append(f"{cid}: video list {meta['video']} contains 01")
    elif meta["video"] != "02": errs.append(f"{cid}: video {meta['video']}")
    lin = np.loadtxt(f"{cdir}/lineage.txt", dtype=np.int64, ndmin=2)
    tracks = {int(L): (int(B), int(E), int(P)) for L, B, E, P in lin}
    mt = np.loadtxt(f"{ctc}/{cid}_GT/TRA/man_track.txt", dtype=np.int64, ndmin=2)
    if not np.array_equal(lin, mt): errs.append(f"{cid}: lineage.txt != man_track.txt")
    mothers, daughters = MC.division_events(tracks)
    # every mother has exactly 2 daughters born at E+1
    for m, E in mothers.items():
        ds_ = [d for d, (B, _, P) in tracks.items() if P == m]
        if len(ds_) != 2 or any(tracks[d][0] != E + 1 for d in ds_):
            errs.append(f"{cid}: mother {m} daughters {ds_}"); stats["lineage_bad"] += 1
    for L, (B, E, P) in tracks.items():
        if P > 0 and (P not in tracks or tracks[P][1] + 1 != B):
            errs.append(f"{cid}: id {L} parent {P} E+1 != B"); stats["lineage_bad"] += 1
        if P > 0 and P not in mothers:
            errs.append(f"{cid}: id {L} parent {P} has <2 children"); stats["lineage_bad"] += 1
    stats["divs"] += len(mothers); stats["ids"] += len(tracks)
    present = {}
    # ref
    rp = f"{cdir}/ref.png"
    with open(rp, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n": stats["bad_png"] += 1; errs.append(cid + " ref not PNG")
    ref = cv2.imread(rp, -1)
    if ref.shape != (H, W, 3) or not (np.array_equal(ref[..., 0], ref[..., 1]) and np.array_equal(ref[..., 1], ref[..., 2])):
        stats["bad_shape"] += 1; errs.append(f"{cid}: ref shape {ref.shape} / not gray")
    ri = meta["ref"]
    if lead:
        mpp = f"{N}/data/vace_clips/{ds}/intensity_mapping.json"
        vid = ri.get("video") or meta["video"]
        if vid not in meta["video"].split(","): errs.append(f"{cid}: ref video {vid} not in {meta['video']}")
        im = tifffile.imread(f"{N}/data/ctc/{ds}/{vid}/t{ri['frame']:03d}.tif")
        if ri["crop"]:
            c = ri["crop"]; im = im[c["y0"]:c["y0"] + c["h"], c["x0"]:c["x0"] + c["w"]]
            if (c["y0"], c["x0"], c["h"], c["w"]) not in LEAD_CROPS: errs.append(f"{cid}: ref crop {c}")
        else:
            errs.append(f"{cid}: Lead-BF ref without crop")
        if os.path.exists(mpp):
            mp = json.load(open(mpp))
            if ri["mapping"] != mpp or ri["lo"] != mp["lo"] or ri["hi"] != mp["hi"]: errs.append(f"{cid}: ref mapping {ri}")
            lo, hi = mp["lo"], mp["hi"]
        else:  # lineage_sim.load_ref falls back to percentiles 1/99.9 of the cropped frame
            lo, hi = float(np.percentile(im, 1.0)), float(np.percentile(im, 99.9))
            if ri["mapping"] != "frame percentiles 1/99.9" or ri["lo"] != lo or ri["hi"] != hi:
                errs.append(f"{cid}: ref mapping {ri} != percentiles {lo},{hi}")
        exp = MC.to_uint8(im, lo, hi)
    else:
        mp = json.load(open(f"{N}/data/vace_clips/{ds}/intensity_mapping.json"))
        if ri["mapping"] != f"{N}/data/vace_clips/{ds}/intensity_mapping.json" or ri["lo"] != mp["lo"] or ri["hi"] != mp["hi"]:
            errs.append(f"{cid}: ref mapping {ri}")
        im = tifffile.imread(f"{N}/data/ctc/{ds}/02/t{ri['frame']:03d}.tif")
        if ri["crop"]:
            c = ri["crop"]; im = im[c["y0"]:c["y0"] + c["h"], c["x0"]:c["x0"] + c["w"]]
            if (c["y0"], c["x0"], c["h"], c["w"]) not in [(94, 0, 512, 768), (94, 166, 512, 768), (94, 332, 512, 768)]:
                errs.append(f"{cid}: ref crop {c}")
        exp = MC.to_uint8(im, mp["lo"], mp["hi"])
    if not np.array_equal(exp, ref[..., 0]): stats["ref_mismatch"] += 1; errs.append(f"{cid}: ref != real frame mapped")
    cdn = meta.get("cond_dir", "cond") if lead else "cond"
    cver = int(meta.get("cond_version", 1)) if lead else 1
    for t in range(nf):
        lp = f"{cdir}/labels/frame_{t:03d}.png"; cp = f"{cdir}/{cdn}/frame_{t:03d}.png"
        for p in (lp, cp):
            with open(p, "rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n": stats["bad_png"] += 1; errs.append(p + " not PNG")
        lab = cv2.imread(lp, -1)
        cond = cv2.imread(cp, -1)[..., ::-1]
        if lab.dtype != np.uint16 or lab.shape != (H, W): stats["bad_shape"] += 1; errs.append(f"{lp}: {lab.dtype} {lab.shape}")
        if cond.shape != (H, W, 3) or cond.dtype != np.uint8: stats["bad_shape"] += 1; errs.append(f"{cp}: {cond.shape}")
        seg = tifffile.imread(f"{ctc}/{cid}_ST/SEG/man_seg{t:03d}.tif")
        tra = tifffile.imread(f"{ctc}/{cid}_GT/TRA/man_track{t:03d}.tif")
        if seg.dtype != np.uint16 or seg.shape != (H, W): stats["bad_shape"] += 1; errs.append(f"{cid} seg {t} {seg.dtype} {seg.shape}")
        if not np.array_equal(seg, lab): stats["label_mismatch"] += 1; errs.append(f"{cid} f{t}: labels png != ST tif")
        if not np.array_equal(np.maximum(tra, seg), seg): stats["tra_outside"] += 1; errs.append(f"{cid} f{t}: TRA marker outside SEG")
        ids = np.unique(lab); ids = ids[ids > 0]
        tids = np.unique(tra); tids = tids[tids > 0]
        if set(ids.tolist()) != set(tids.tolist()): stats["tra_missing"] += 1; errs.append(f"{cid} f{t}: TRA ids != SEG ids")
        for lid in ids:
            lid = int(lid)
            present.setdefault(lid, []).append(t)
            m = lab == lid
            if ndi.label(m)[1] != 1: stats["multi_cc4"] += 1; errs.append(f"{cid} f{t} id {lid}: {ndi.label(m)[1]} 4-cc")
            if ndi.label(m, structure=np.ones((3, 3)))[1] != 1: stats["multi_cc8"] += 1
            if ndi.label(tra == lid, structure=np.ones((3, 3)))[1] != 1: stats["tra_multi"] += 1
        # cond byte-compat with make_clips encoder (lineage is clip-local so t_global = t)
        if lead:
            exp = CR.cond_image(lab, tracks, t, cver, mothers, daughters)
        else:
            exp = MC.cond_image(lab, t, mothers, daughters)
        d = int((exp != cond).any(-1).sum())
        if d: stats["cond_mismatch_px"] += d; errs.append(f"{cid} f{t}: cond differs from make_clips encoder in {d} px")
        stats["mother_marked_frames"] += int((cond[..., 2] == 255).any()); stats["daughter_marked_frames"] += int((cond[..., 2] == 128).any())
        if not set(np.unique(cond[..., 2]).tolist()) <= {0, 128, 255}: errs.append(f"{cid} f{t}: B values {np.unique(cond[...,2])}")
        if not set(np.unique(cond[..., 0]).tolist()) <= {0, 255}: errs.append(f"{cid} f{t}: R values")
        if lead and cver == 2:
            sep = CR.separation_mask(lab)
            if not np.array_equal(cond[..., 1] > 0, (lab > 0) & ~sep): errs.append(f"{cid} f{t}: G support != foreground minus separation line")
            if not (cond[..., 0][sep] == 255).all(): errs.append(f"{cid} f{t}: R != 255 on separation line")
            if cond[..., 2].any(): errs.append(f"{cid} f{t}: B != 0 in cond_v2")
        elif not np.array_equal(cond[..., 1] > 0, lab > 0): errs.append(f"{cid} f{t}: G support != foreground")
        stats["cells"].append(len(ids)); stats["frames"] += 1
    for lid, (B, E, P) in tracks.items():
        if present.get(lid) != list(range(B, E + 1)): stats["lineage_bad"] += 1; errs.append(f"{cid}: id {lid} present {present.get(lid)} vs [{B},{E}]")
    if set(present) != set(tracks): stats["lineage_bad"] += 1; errs.append(f"{cid}: ids in masks != lineage")
    if meta["n_div"] != len(mothers): errs.append(f"{cid}: meta n_div {meta['n_div']} != {len(mothers)}")
    stats["clips"] += 1
stats["mean_cells_per_frame"] = float(np.mean(stats.pop("cells")))
print(json.dumps(stats, indent=1))
print("ERRORS:", len(errs))
for e in errs[:40]: print(" ", e)
