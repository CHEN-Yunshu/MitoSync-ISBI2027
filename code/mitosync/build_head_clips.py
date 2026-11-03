#!/usr/bin/env python
"""
build_head_clips.py -- head-only VACE LoRA training clip set (MitoSync protocol, 2026-10-10).

Mirrors the v2 LoRA training set (data/vace_clips_div/<ds>/metadata_div.csv: 17-frame clips of video 02 at start
stride 2, >= 1 daughter birth in clip frames 5..11, conditioning cond_v2/, cross-clip reference = ref.png of a
DIFFERENT kept clip of the same crop position drawn with random.Random(0), exactly as make_div_set.py) but keeps a
row only if
  (a) its 17-frame window lies entirely in the 02 head [0, head_last] (head_last = last frame of
      data/ctc/<ds>_trainhead/02, verified against the md5 of the original frames), and
  (b) its cond_v2 event channel needs no tail information: cond_render marks a mother in her last 3 frames E-2..E;
      a mother with E >= head_last has her daughters born in the tail -> such a clip is dropped.  As an independent
      check the clip's event marks are recomputed from the HEAD-ONLY lineage (data/ctc/<ds>_trainhead/02_GT/TRA/
      man_track.txt) and must be identical to the marks rendered from the full-02 lineage.
Reference images are re-drawn among the kept head clips only (same rule and seed as make_div_set.py).
Clip dirs are SYMLINKED to the existing clip dirs (no image copies); nothing existing is modified.

Optionally (--tail_view DIR) writes an EVALUATION-ONLY gen_eval view of clips that overlap the 02 tail
(cond -> cond_v2, labels, lineage.txt, meta.json, target symlinked; ref.png -> ref.png of the latest kept HEAD clip of
the same crop position), used only for checkpoint selection / gating of the head LoRA on the tail.

usage (vace venv):
  build_head_clips.py --dataset PhC-C2DL-PSC [--tail_view $N/eval/mitosync/lora_head_gate/PhC-C2DL-PSC/clips_tail_view
                       --tail_min_frames 17 --tail_n 8]
"""
import argparse, csv, glob, hashlib, json, os, random, re, sys
import numpy as np
from PIL import Image

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))


def load_man_track(path):
    arr = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return {int(L): (int(B), int(E), int(P)) for L, B, E, P in arr}


def division_events(tracks):
    """identical to make_clips.division_events: mothers {mid: E} (>= 2 children), daughters {did: B}."""
    children = {}
    for L, (B, E, P) in tracks.items():
        if P > 0:
            children.setdefault(P, []).append(L)
    mothers, daughters = {}, {}
    for P, ch in children.items():
        if len(ch) >= 2 and P in tracks:
            mothers[P] = tracks[P][1]
            for d in ch:
                daughters[d] = tracks[d][0]
    return mothers, daughters


def clip_marks(label_frames, frames_global, mothers, daughters):
    """event marks exactly as cond_render.event_channel: (id, local t, 'M'|'D')."""
    marks = set()
    for tl, (lab, g) in enumerate(zip(label_frames, frames_global)):
        ids = set(np.unique(lab).tolist()) - {0}
        for lid in ids:
            if lid in mothers and mothers[lid] - 2 <= g <= mothers[lid]:
                marks.add((lid, tl, "M"))
            elif lid in daughters and daughters[lid] == g:
                marks.add((lid, tl, "D"))
    return marks


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--src", default=None, help="default $N/data/vace_clips_div/<ds>")
    ap.add_argument("--src_csv", default="metadata_div.csv")
    ap.add_argument("--out", default=None, help="default $N/data/vace_clips_head/<ds>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail_view", default=None)
    ap.add_argument("--tail_min_frames", type=int, default=17)
    ap.add_argument("--tail_n", type=int, default=8)
    a = ap.parse_args()
    ds = a.dataset
    src = a.src or f"{N}/data/vace_clips_div/{ds}"
    out = a.out or f"{N}/data/vace_clips_head/{ds}"
    ctc = f"{N}/data/ctc"

    # ---- head split, verified against the original frames
    head_frames = sorted(glob.glob(f"{ctc}/{ds}_trainhead/02/t*.tif"))
    tail_frames = sorted(glob.glob(f"{ctc}/{ds}_valtail/02/t*.tif"))
    full_frames = sorted(glob.glob(f"{ctc}/{ds}/02/t*.tif"))
    head_last = len(head_frames) - 1
    assert len(head_frames) + len(tail_frames) == len(full_frames), (len(head_frames), len(tail_frames), len(full_frames))
    for i in (0, head_last):
        assert md5(head_frames[i]) == md5(f"{ctc}/{ds}/02/t{i:03d}.tif"), ("head frame mismatch", i)
    assert md5(tail_frames[0]) == md5(f"{ctc}/{ds}/02/t{head_last + 1:03d}.tif"), "tail[0] != full[head_last+1]"
    print(f"[head] {ds}: full 02 = {len(full_frames)} frames, head = 0..{head_last} ({len(head_frames)}), "
          f"tail = {head_last + 1}..{len(full_frames) - 1} ({len(tail_frames)}) [md5-verified]")

    tracks_full = load_man_track(f"{ctc}/{ds}/02_GT/TRA/man_track.txt")
    tracks_head = load_man_track(f"{ctc}/{ds}_trainhead/02_GT/TRA/man_track.txt")
    m_full, d_full = division_events(tracks_full)
    m_head, d_head = division_events(tracks_head)

    rows = list(csv.DictReader(open(os.path.join(src, a.src_csv))))
    header = ["video", "vace_video", "vace_reference_image", "prompt"]
    sdiv = json.load(open(os.path.join(src, "summary_div.json")))
    info, kept, dropped = {}, [], []
    for r in rows:
        c = r["video"].split("/")[0]
        cond_dir = r["vace_video"].split("/")[1]
        meta = json.load(open(os.path.join(src, c, "meta.json")))
        fg = meta["frames_global"]
        m = re.match(r"clip_(\d+)(?:_(\w+))?$", c)
        crop = m.group(2) or "full"
        rec = dict(clip=c, start=meta["start_frame"], last=fg[-1], crop=crop, cond_dir=cond_dir, prompt=r["prompt"],
                   div_events_win=sdiv["clips"][c]["n_div_events_win"], daughters_win=sdiv["clips"][c]["n_daughters_win"])
        info[c] = rec
        if fg[-1] > head_last:
            rec["reason"] = "window_not_in_head"; dropped.append(rec); continue
        labs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(os.path.join(src, c, "labels", "*.png")))]
        assert len(labs) == len(fg) == 17, (c, len(labs))
        mk_full = clip_marks(labs, fg, m_full, d_full)
        mk_head = clip_marks(labs, fg, m_head, d_head)
        tail_mothers = sorted({lid for (lid, tl, k) in mk_full if k == "M" and m_full[lid] >= head_last})
        rec.update(n_marks=len(mk_full), tail_mothers=tail_mothers, marks_equal_head_lineage=(mk_full == mk_head))
        if tail_mothers:
            rec["reason"] = f"cond_marks_mother_dividing_in_tail {tail_mothers}"; dropped.append(rec); continue
        assert mk_full == mk_head, (c, sorted(mk_full ^ mk_head)[:10])
        kept.append(c)

    # ---- cross-clip refs among HEAD clips only (make_div_set.py rule, same seed)
    rng = random.Random(a.seed)
    by_crop = {}
    for c in kept:
        by_crop.setdefault(info[c]["crop"], []).append(c)
    refs = {}
    for c in kept:
        pool = [x for x in by_crop[info[c]["crop"]] if x != c]
        refs[c] = rng.choice(pool) if pool else c
    assert all(info[refs[c]]["last"] <= head_last for c in kept)

    os.makedirs(out, exist_ok=True)
    for c in kept:
        dst = os.path.join(out, c)
        if not os.path.islink(dst):
            os.symlink(os.path.realpath(os.path.join(src, c)), dst)
        assert os.path.realpath(dst) == os.path.realpath(os.path.join(src, c))
    with open(os.path.join(out, "metadata_div.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(header)
        for c in kept:
            wr.writerow([f"{c}/target", f"{c}/{info[c]['cond_dir']}", f"{refs[c]}/ref.png", info[c]["prompt"]])
    with open(os.path.join(out, "metadata.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(header)
        for c in kept:
            wr.writerow([f"{c}/target", f"{c}/{info[c]['cond_dir']}", f"{c}/ref.png", info[c]["prompt"]])
    im_src = os.path.join(src, "intensity_mapping.json")
    if not os.path.islink(os.path.join(out, "intensity_mapping.json")):
        os.symlink(os.path.realpath(im_src), os.path.join(out, "intensity_mapping.json"))

    # ---- head-only intensity percentiles vs the mapping the PNGs were rendered with (full 02)
    hist = None
    for fpath in head_frames:
        import tifffile
        im = tifffile.imread(fpath)
        if hist is None:
            hist = np.zeros(int(np.iinfo(im.dtype).max) + 1, np.int64)
        hist += np.bincount(im.ravel().astype(np.int64), minlength=hist.size)
    cdf = np.cumsum(hist); n = int(cdf[-1])
    pct = lambda p: int(np.searchsorted(cdf, max(1, int(np.ceil(p / 100.0 * n)))))
    imap = json.load(open(im_src))
    icheck = dict(mapping_used_by_pngs=dict(lo=imap["lo"], hi=imap["hi"], computed_over="all frames of 02"),
                  head_only=dict(lo=pct(imap["p_low"]), hi=pct(imap["p_high"]), computed_over=f"frames 0..{head_last} of 02"),
                  identical=(pct(imap["p_low"]) == imap["lo"] and pct(imap["p_high"]) == imap["hi"]))
    json.dump(icheck, open(os.path.join(out, "head_intensity_check.json"), "w"), indent=1)

    # ---- summary
    full_rows = len(rows)
    win_head = [r for r in info.values() if r["last"] <= head_last]
    summ = dict(dataset=ds, protocol="MitoSync head-only (2026-10-10)", src=src, src_csv=a.src_csv, out=out,
                head_last=head_last, n_frames_full=len(full_frames), n_rows_full_v2=full_rows,
                n_window_in_head=len(win_head), n_dropped_lookahead=sum(1 for r in dropped if r["reason"].startswith("cond_marks")),
                n_kept=len(kept), kept_per_crop={k: len(v) for k, v in sorted(by_crop.items())},
                starts_kept=[min(info[c]["start"] for c in kept), max(info[c]["start"] for c in kept)],
                div_events_win_total_kept=int(sum(info[c]["div_events_win"] for c in kept)),
                div_events_win_total_full=int(sum(r["div_events_win"] for r in info.values())),
                daughters_win_total_kept=int(sum(info[c]["daughters_win"] for c in kept)),
                daughters_win_total_full=int(sum(r["daughters_win"] for r in info.values())),
                ref_seed=a.seed, marks_equal_head_lineage_all_kept=all(info[c]["marks_equal_head_lineage"] for c in kept),
                intensity_check=icheck,
                metadata_div_sha256=sha256(os.path.join(out, "metadata_div.csv")),
                metadata_sha256=sha256(os.path.join(out, "metadata.csv")),
                kept=[dict(clip=c, start=info[c]["start"], crop=info[c]["crop"], ref=refs[c]) for c in kept],
                dropped=[dict(clip=r["clip"], start=r["start"], reason=r["reason"]) for r in dropped])

    # ---- evaluation-only tail view
    if a.tail_view:
        ntail = {c: sum(1 for g in json.load(open(os.path.join(src, c, "meta.json")))["frames_global"] if g > head_last)
                 for c in info}
        cand = sorted([c for c in info if ntail[c] >= a.tail_min_frames], key=lambda c: (info[c]["start"], c))
        starts = sorted(set(info[c]["start"] for c in cand))
        ncrop = max(1, len(by_crop))
        max_starts = max(1, a.tail_n // ncrop)
        if len(starts) > max_starts:
            idx = np.round(np.linspace(0, len(starts) - 1, max_starts)).astype(int)
            starts = [starts[i] for i in sorted(set(idx.tolist()))]
        sel = [c for c in cand if info[c]["start"] in starts]
        latest_head = {k: max(v, key=lambda c: info[c]["start"]) for k, v in by_crop.items()}
        os.makedirs(a.tail_view, exist_ok=True)
        tv = []
        for c in sel:
            d = os.path.join(a.tail_view, c); os.makedirs(d, exist_ok=True)
            links = {"cond": os.path.join(src, c, info[c]["cond_dir"]), "labels": os.path.join(src, c, "labels"),
                     "lineage.txt": os.path.join(src, c, "lineage.txt"), "meta.json": os.path.join(src, c, "meta.json"),
                     "target": os.path.join(src, c, "target"),
                     "ref.png": os.path.join(src, latest_head[info[c]["crop"]], "ref.png")}
            for k, v in links.items():
                if not os.path.islink(os.path.join(d, k)):
                    os.symlink(os.path.realpath(v), os.path.join(d, k))
            tv.append(dict(clip=c, start=info[c]["start"], crop=info[c]["crop"], tail_frames=ntail[c],
                           ref_from_head_clip=latest_head[info[c]["crop"]], div_events_win=info[c]["div_events_win"]))
        summ["tail_view"] = dict(dir=a.tail_view, purpose="EVALUATION ONLY (checkpoint selection / gate on the 02 tail)",
                                 min_tail_frames=a.tail_min_frames, clips=tv)
        json.dump(summ["tail_view"], open(os.path.join(a.tail_view, "tail_view.json"), "w"), indent=1)

    json.dump(summ, open(os.path.join(out, "summary_head.json"), "w"), indent=1)
    lines = [f"{ds}: head 0..{head_last} of {len(full_frames)}; v2 full-02 rows {full_rows}; window-in-head {len(win_head)}; "
             f"dropped by cond look-ahead {summ['n_dropped_lookahead']}; KEPT {len(kept)} {summ['kept_per_crop']} starts {summ['starts_kept']}; "
             f"div events in window kept/full {summ['div_events_win_total_kept']}/{summ['div_events_win_total_full']}; "
             f"intensity mapping head-identical={icheck['identical']} (full {icheck['mapping_used_by_pngs']['lo']}/{icheck['mapping_used_by_pngs']['hi']}, "
             f"head {icheck['head_only']['lo']}/{icheck['head_only']['hi']}); metadata_div sha256 {summ['metadata_div_sha256']}"]
    for r in dropped:
        if r["reason"].startswith("cond_marks"):
            lines.append(f"  dropped {r['clip']} start {r['start']}: {r['reason']}")
    if a.tail_view:
        lines.append(f"  tail view ({len(summ['tail_view']['clips'])} clips): " +
                     ", ".join(f"{t['clip']}(s{t['start']},{t['tail_frames']}tail,ref {t['ref_from_head_clip']})" for t in summ["tail_view"]["clips"]))
    open(os.path.join(out, "summary_head.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
