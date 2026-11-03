#!/usr/bin/env python
"""Independent verifier for relabel_by_readout.py outputs (written by the verifier, not the builder).

usage: python relabel_verify_indep.py <orig_ctc_root> <rl_root> [clip names...]   (no names = all clips)
Per clip: own IoU>=0.5 one-to-one greedy matching from cpsam.npz, own division outcome (kept/delayed/dropped),
compared with relabel_summary.json; pixel-level audit of every changed pixel; track-table audit; byte-identity of
non-division tracks; ctc_io.verify_ctc + indep_ctc_check.check via a temp 02-layout symlink tree.
"""
import os, sys, json, glob, tempfile, hashlib
from collections import defaultdict
import numpy as np, tifffile
from scipy import ndimage as ndi
from pathlib import Path

sys.path.insert(0, "STUDY_ROOT/scripts/vace")
from ctc_io import verify_ctc  # noqa
import indep_ctc_check  # noqa

S4 = ndi.generate_binary_structure(2, 1)


def tracks(p):
    t = np.loadtxt(p, dtype=int).reshape(-1, 4)
    return {int(L): (int(B), int(E), int(P)) for L, B, E, P in t}


def my_match(lab, pred, thr=0.5):
    """dense IoU matrix, greedy descending, one-to-one. returns set of matched label ids, n_lab, n_pred"""
    ids_l = [i for i in np.unique(lab) if i]; ids_p = [i for i in np.unique(pred) if i]
    if not ids_l or not ids_p:
        return set(), len(ids_l), len(ids_p)
    il = {v: k for k, v in enumerate(ids_l)}; ip = {v: k for k, v in enumerate(ids_p)}
    inter = np.zeros((len(ids_l), len(ids_p)), np.int64)
    both = (lab > 0) & (pred > 0)
    la = np.array([il[a] for a in ids_l])  # noqa (identity)
    LI = np.zeros(int(lab.max()) + 1, np.int64); LI[ids_l] = np.arange(len(ids_l))
    PI = np.zeros(int(pred.max()) + 1, np.int64); PI[ids_p] = np.arange(len(ids_p))
    np.add.at(inter, (LI[lab[both]], PI[pred[both]]), 1)
    al = np.array([(lab == i).sum() for i in ids_l]); ap = np.array([(pred == i).sum() for i in ids_p])
    iou = inter / (al[:, None] + ap[None, :] - inter)
    matched = set(); used_p = set()
    flat = [(iou[i, j], i, j) for i in range(len(ids_l)) for j in range(len(ids_p)) if iou[i, j] >= thr]
    flat.sort(key=lambda x: -x[0])
    for v, i, j in flat:
        if ids_l[i] in matched or j in used_p:
            continue
        matched.add(ids_l[i]); used_p.add(j)
    return matched, len(ids_l), len(ids_p)


def f1(m, nl, np_):
    tp = len(m)
    if nl == 0 and np_ == 0:
        return 1.0
    return 2 * tp / max(2 * tp + (np_ - tp) + (nl - tp), 1)


def check_clip(orig_root, rl_root, name, summ):
    out = dict(clip=name, problems=[])
    P_ = out["problems"]
    T = len(glob.glob(f"{orig_root}/{name}/t*.tif"))
    seg_o = np.stack([tifffile.imread(f"{orig_root}/{name}_ST/SEG/man_seg{t:03d}.tif").astype(np.int32) for t in range(T)])
    tra_o = np.stack([tifffile.imread(f"{orig_root}/{name}_GT/TRA/man_track{t:03d}.tif").astype(np.int32) for t in range(T)])
    seg_r = np.stack([tifffile.imread(f"{rl_root}/{name}_ST/SEG/man_seg{t:03d}.tif").astype(np.int32) for t in range(T)])
    tra_r = np.stack([tifffile.imread(f"{rl_root}/{name}_GT/TRA/man_track{t:03d}.tif").astype(np.int32) for t in range(T)])
    tr_o = tracks(f"{orig_root}/{name}_GT/TRA/man_track.txt"); tr_r = tracks(f"{rl_root}/{name}_GT/TRA/man_track.txt")
    pred = np.load(f"{rl_root}/{name}/cpsam.npz")["masks"].astype(np.int32)
    assert pred.shape == seg_o.shape == seg_r.shape
    # images are symlinks to originals?
    for t in range(T):
        p = f"{rl_root}/{name}/t{t:03d}.tif"
        if os.path.realpath(p) != os.path.realpath(f"{orig_root}/{name}/t{t:03d}.tif"):
            P_.append(f"image t{t:03d} not symlink to original")
    # own matching
    M = []; f1o = []; f1n = []
    for t in range(T):
        m, nl, npd = my_match(seg_o[t], pred[t]); M.append(m); f1o.append(f1(m, nl, npd))
        m2, nl2, npd2 = my_match(seg_r[t], pred[t]); f1n.append(f1(m2, nl2, npd2))
    # divisions from ORIGINAL tracks
    ch = defaultdict(list)
    for L, (B, E, P) in tr_o.items():
        if P:
            ch[P].append(L)
    divs = {}
    for P, c in ch.items():
        if len(c) == 2:
            d1, d2 = sorted(c)
            tb = tr_o[d1][0]
            if tr_o[d2][0] != tb or tr_o[P][1] != tb - 1:
                divs[P] = dict(d=(d1, d2), tb=tb, outcome="skipped_irregular"); continue
            ts = None
            for t in range(tb, T - 1):
                if d1 in M[t] and d2 in M[t] and d1 in M[t + 1] and d2 in M[t + 1]:
                    ts = t; break
            if ts == tb:
                oc = "kept"
            elif ts is None:
                oc = "dropped"
            else:
                oc = "delayed"
            divs[P] = dict(d=(d1, d2), tb=tb, t_star=ts, outcome=oc)
    # compare with builder's records
    recs = {r["parent"]: r for r in summ["divisions"]}
    mism = []
    for P, d in divs.items():
        r = recs.get(P)
        if r is None:
            mism.append((P, "missing in summary")); continue
        if r["outcome"] != d["outcome"] or (d["outcome"] == "delayed" and r.get("t_star") != d["t_star"]):
            mism.append((P, d["outcome"], d.get("t_star"), r["outcome"], r.get("t_star")))
    if set(recs) != set(divs):
        mism.append(("division set differs", sorted(set(recs) ^ set(divs))))
    out["division_mismatches"] = mism
    cnt = defaultdict(int)
    for d in divs.values():
        cnt[d["outcome"]] += 1
    out["my_counts"] = dict(cnt)
    out["my_delays"] = sorted(d["t_star"] - d["tb"] for d in divs.values() if d["outcome"] == "delayed")
    out["builder_counts"] = dict(kept=summ["kept_at_birth"], delayed=summ["delayed"], dropped=summ["dropped"])
    out["builder_delays"] = sorted(summ["delays"])
    out["my_f1_orig"], out["my_f1_new"] = float(np.mean(f1o)), float(np.mean(f1n))
    out["builder_f1_orig"], out["builder_f1_new"] = summ["readout_f1_orig"], summ["readout_f1_new"]
    if abs(out["my_f1_orig"] - summ["readout_f1_orig"]) > 1e-6 or abs(out["my_f1_new"] - summ["readout_f1_new"]) > 1e-6:
        P_.append("F1 mismatch")
    # ---- expected relabel map: (t, old_id) -> new_id for daughters of delayed/dropped divisions
    remap = {}  # (t, id) -> mother
    mod_ids = set()
    exp_tr = {L: list(v) for L, v in tr_o.items()}
    for P, d in divs.items():
        d1, d2 = d["d"]; tb = d["tb"]
        if d["outcome"] == "delayed":
            ts = d["t_star"]
            for t in range(tb, ts):
                remap[(t, d1)] = P; remap[(t, d2)] = P
            exp_tr[P][1] = ts - 1; exp_tr[d1][0] = ts; exp_tr[d2][0] = ts
            mod_ids |= {P, d1, d2}
        elif d["outcome"] == "dropped":
            Emax = max(tr_o[d1][1], tr_o[d2][1])
            for t in range(tb, Emax + 1):
                remap[(t, d1)] = P; remap[(t, d2)] = P
            exp_tr[P][1] = max(exp_tr[P][1], Emax)
            exp_tr.pop(d1); exp_tr.pop(d2)
            mod_ids |= {P, d1, d2}
            # children of dropped daughters (sim: none) -> reparent
            for L, v in exp_tr.items():
                if v[2] in (d1, d2):
                    v[2] = P
    # track table check (a)/(b)
    exp_rows = {L: tuple(v) for L, v in exp_tr.items()}
    if exp_rows != tr_r:
        diff = {L: (exp_rows.get(L), tr_r.get(L)) for L in set(exp_rows) | set(tr_r) if exp_rows.get(L) != tr_r.get(L)}
        P_.append(f"man_track.txt differs from expectation: {diff}")
    # dropped daughters absent anywhere (b)
    dropped_ids = {x for P, d in divs.items() if d["outcome"] == "dropped" for x in d["d"]}
    present_r = set(np.unique(seg_r)) | set(np.unique(tra_r))
    if dropped_ids & present_r:
        P_.append(f"dropped daughter ids still present: {dropped_ids & present_r}")
    # pixel audit
    px_relabel = 0; px_bridge = 0; px_frag_bg = 0; px_foreign = 0; px_unexplained = 0; frames_changed = 0
    unexplained_ex = []
    for t in range(T):
        o, r = seg_o[t], seg_r[t]
        if (o != r).any():
            frames_changed += 1
        expected = o.copy()
        for (tt, i), m in remap.items():
            if tt == t:
                expected[o == i] = m
        px_relabel += int(((expected != o)).sum())
        diff = expected != r
        if not diff.any():
            continue
        mothers_t = {m for (tt, i), m in remap.items() if tt == t}
        # bridge: expected background -> mother
        b = diff & (expected == 0) & np.isin(r, list(mothers_t))
        px_bridge += int(b.sum())
        # fragment to bg: expected mother -> 0
        fb = diff & np.isin(expected, list(mothers_t)) & (r == 0)
        px_frag_bg += int(fb.sum())
        # foreign: expected other cell -> mother
        fo = diff & (expected > 0) & np.isin(r, list(mothers_t))  # any other cell's pixel taken by a mother (foreign crossing)
        px_foreign += int(fo.sum())
        rest = diff & ~b & ~fb & ~fo
        if rest.any():
            px_unexplained += int(rest.sum())
            ys, xs = np.nonzero(rest)
            unexplained_ex.append((t, int(ys[0]), int(xs[0]), int(expected[ys[0], xs[0]]), int(r[ys[0], xs[0]])))
        # bridge pixels must be adjacent-ish: all mothers one 4-component in rl
        for m in mothers_t:
            if (r == m).any() and ndi.label(r == m, structure=S4)[1] != 1:
                P_.append(f"t{t} mother {m} not one 4-component")
    out.update(my_frames_changed=frames_changed, my_px_relabel=px_relabel, my_px_bridge=px_bridge, my_px_frag_bg=px_frag_bg,
               my_px_foreign=px_foreign, my_px_unexplained=px_unexplained, unexplained_examples=unexplained_ex[:5],
               builder=dict(frames_relabelled=summ["frames_relabelled"], pixels_relabelled=summ["pixels_relabelled"],
                            bridge_pixels_added=summ["bridge_pixels_added"], frag_bg=summ["bridge_fragment_to_bg_pixels"],
                            foreign=summ["bridge_foreign_pixels"]))
    # note: builder's pixels_relabelled counts pixels of daughters at merge time (after earlier merges) == expected!=o count
    if px_relabel != summ["pixels_relabelled"]:
        P_.append(f"pixels_relabelled {px_relabel} != builder {summ['pixels_relabelled']}")
    if px_bridge != summ["bridge_pixels_added"]:
        P_.append(f"bridge px {px_bridge} != builder {summ['bridge_pixels_added']}")
    if px_frag_bg != summ["bridge_fragment_to_bg_pixels"]:
        P_.append(f"frag_bg px {px_frag_bg} != builder {summ['bridge_fragment_to_bg_pixels']}")
    if px_foreign != summ["bridge_foreign_pixels"]:
        P_.append(f"foreign px {px_foreign} != builder {summ['bridge_foreign_pixels']}")
    if frames_changed != summ["frames_relabelled"]:
        P_.append(f"frames changed {frames_changed} != builder {summ['frames_relabelled']}")
    # (c) non-division tracks byte-identical: masks + markers + track rows
    other = set(tr_o) - mod_ids
    nd_bad_mask = nd_bad_marker = 0
    for i in other:
        if ((seg_o == i) != (seg_r == i)).any():
            nd_bad_mask += 1
        if ((tra_o == i) != (tra_r == i)).any():
            nd_bad_marker += 1
        if tr_o[i] != tr_r.get(i):
            P_.append(f"non-division track row changed {i}: {tr_o[i]} -> {tr_r.get(i)}")
    out.update(non_division_ids=len(other), non_division_mask_changed=nd_bad_mask, non_division_marker_changed=nd_bad_marker)
    # modified ids outside modified frames identical (e.g. daughters after t*, mother before tb)
    mod_bad = 0
    for P, d in divs.items():
        if d["outcome"] not in ("delayed", "dropped"):
            continue
        d1, d2 = d["d"]; tb = d["tb"]
        for t in range(0, tb):
            if ((seg_o[t] == P) != (seg_r[t] == P)).any():
                mod_bad += 1
        if d["outcome"] == "delayed":
            for t in range(d["t_star"], T):
                for i in (d1, d2):
                    if ((seg_o[t] == i) != (seg_r[t] == i)).any():
                        mod_bad += 1
    out["modified_ids_changed_outside_window"] = mod_bad
    # markers: every rl marker inside its own rl mask, one component per id, disc-ish
    mk_bad = 0
    for t in range(T):
        m = tra_r[t]; s = seg_r[t]
        if ((m > 0) & (s != m)).any():
            mk_bad += 1
    out["marker_outside_mask_frames"] = mk_bad
    # (d) verify_ctc + indep_ctc_check
    v = verify_ctc(f"{rl_root}/{name}")
    out["verify_ctc"] = {k: v[k] for k in ("bad_frames", "bad_parents", "missing_ids", "extra_ids", "multi_component", "divisions") if k in v}
    with tempfile.TemporaryDirectory(dir="USER_DATA/tmp") as td:
        os.makedirs(f"{td}/{name}")
        os.symlink(f"{rl_root}/{name}", f"{td}/{name}/02"); os.symlink(f"{rl_root}/{name}_GT", f"{td}/{name}/02_GT"); os.symlink(f"{rl_root}/{name}_ST", f"{td}/{name}/02_ST")
        indep_ctc_check.D = Path(td)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            indep_ctc_check.check(name)
        out["indep_ctc_check"] = json.loads(buf.getvalue().strip().splitlines()[-1])
    return out


def main():
    orig_root, rl_root = sys.argv[1], sys.argv[2]
    S = json.load(open(f"{rl_root}/relabel_summary.json"))["clips"]
    names = sys.argv[3:] or sorted(S)
    agg = defaultdict(int); bad = []
    for n in names:
        o = check_clip(orig_root, rl_root, n, S[n])
        ic = o["indep_ctc_check"]
        ic_bad = ic["bad_parent_links"] or ic["ids_outside_BE"] or ic["ids_missing_in_both"] or ic["marker_on_other_mask_px"] or not ic["label_fits_uint16"]
        v_bad = any(o["verify_ctc"][k] for k in ("bad_frames", "bad_parents", "missing_ids", "extra_ids", "multi_component"))
        ok = not o["problems"] and not o["division_mismatches"] and o["my_px_unexplained"] == 0 and not ic_bad and not v_bad \
            and o["non_division_mask_changed"] == 0 and o["non_division_marker_changed"] == 0 and o["modified_ids_changed_outside_window"] == 0 and o["marker_outside_mask_frames"] == 0
        for k in ("kept", "delayed", "dropped"):
            agg[k] += o["my_counts"].get(k, 0)
        agg["px_relabel"] += o["my_px_relabel"]; agg["px_bridge"] += o["my_px_bridge"]; agg["px_frag_bg"] += o["my_px_frag_bg"]; agg["px_foreign"] += o["my_px_foreign"]
        agg["frames_changed"] += o["my_frames_changed"]; agg["clips"] += 1; agg["ok"] += int(ok)
        agg["non_div_mask_changed"] += o["non_division_mask_changed"]; agg["non_div_marker_changed"] += o["non_division_marker_changed"]
        if len(names) <= 5:
            print(json.dumps(o, default=str, indent=1))
        else:
            print(n, "OK" if ok else "BAD", o["my_counts"], "f1", round(o["my_f1_orig"], 4), round(o["my_f1_new"], 4),
                  "px", o["my_px_relabel"], o["my_px_bridge"], o["my_px_frag_bg"], o["my_px_foreign"], "unexpl", o["my_px_unexplained"],
                  "nd_changed", o["non_division_mask_changed"], o["non_division_marker_changed"], "probs", o["problems"][:3], o["division_mismatches"][:3], flush=True)
        if not ok:
            bad.append(n)
    print("AGG", json.dumps(dict(agg)), "BAD", bad)


if __name__ == "__main__":
    main()
