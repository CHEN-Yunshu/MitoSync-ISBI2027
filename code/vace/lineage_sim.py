#!/usr/bin/env python
"""
lineage_sim.py -- lineage event simulator: programs NEW identity-mask videos (new layouts, new
division times, contacts, density) from the statistics of real CTC video 02 (CPU only).

Usage (cellres env or vace venv; needs numpy, scipy, tifffile, cv2, skimage):
  python lineage_sim.py --dataset PhC-C2DL-PSC --video 02 --num_clips 40 --num_frames 17 \
      --divisions_per_clip_range 1,3 --contact_prob 0.3 --density_scale 1.0 --seed 0 \
      --out $N/data/vace_sim/PhC-C2DL-PSC/sim_v1 [--n_cells_range a,b] [--fig $N/figs/vace_sim_PhC-C2DL-PSC.png]
  --video may be a comma-separated list (Lead-BF: "low_A1_10_10,low_A1_10_11,mid_A1_10_11"): the trajectory /
  division libraries of all listed videos are merged (ids offset per video, provenance kept in lib["provenance"]),
  ref.png is a random crop of any listed video (meta ref.video), cache _library_<videos joined by '+'>.pkl.

Output layout (identical to make_clips.py real clips, minus target/):
  <out>/<clip_id>/labels/frame_%03d.png   uint16 identity labels (one 4-connected component per id per frame)
  <out>/<clip_id>/cond/frame_%03d.png     RGB lossless: R = 1-px inner contour (255), G = per-instance normalised EDT,
                                           B = mother pixels in the 3 frames before division = 255, daughters at birth = 128
                                           (--cond_version 1, default; --cond_version 2 writes cond_v2/ instead, see cond_render.py)
  <out>/<clip_id>/lineage.txt             L B E P, clip-local frames 0..num_frames-1
  <out>/<clip_id>/ref.png                 random real video-02 frame (8-bit gray x3, dataset intensity_mapping.json of
                                           vace_clips; Fluo: one of the 3 fixed 512x768 crops), recorded in meta.json
  <out>/<clip_id>/meta.json
  <out>_ctc/<clip_id>_GT/TRA/man_track%03d.tif + man_track.txt,  <out>_ctc/<clip_id>_ST/SEG/man_seg%03d.tif
  <out>/summary.json, <out>/summary.txt
Design: trajectory library (all 02 tracks as per-frame (centroid, mask crop); sampled with random D4 transform,
random start offset, optional time reversal), division templates (mother last 3 frames + both daughters' first 3
frames relative to the mother centroid, spliced onto a sampled pre-division segment; daughters continue with real
post-division daughter segments), collision handling (position search minimising overlap, contested pixels to the
nearer centroid, one component per id, area checks with undo), density_scale x real cells/frame average.
"""
import argparse
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
from skimage.segmentation import find_boundaries

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cond_render import cond_image as render_cond, cond_dirname  # noqa: E402

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
CANVAS = {"PhC-C2DL-PSC": (576, 720), "Fluo-N2DL-HeLa": (512, 768), "DIC-C2DH-HeLa": (512, 512),
          "Lead-BF": (576, 720)}
# fixed crops used by make_clips.py for the real clips (ref.png is taken from one of them)
REF_CROPS = {"PhC-C2DL-PSC": [None], "DIC-C2DH-HeLa": [None],
             "Fluo-N2DL-HeLa": [(94, 0, 512, 768, "left"), (94, 166, 512, 768, "centre"), (94, 332, 512, 768, "right")],
             # Lead-BF: 2048x2448 frames, 3x3 grid of native-resolution 576x720 crops, row-major r0c0..r2c2
             "Lead-BF": [(y0, x0, 576, 720, f"r{i}c{j}") for i, y0 in enumerate((0, 736, 1472))
                         for j, x0 in enumerate((0, 864, 1728))]}
# Lead-BF marker radius: median equivalent diameter 2*sqrt(area/pi) of man_seg000 instances = 65.0 (low_A1_10_10) /
# 70.7 (mid_A1_10_11) / 68.9 (high_A1_10_10) px vs PSC 14.5, Fluo 27.5, DIC 119.3 -> MARKER_R interpolated linearly
# between Fluo (27.5 -> 4) and DIC (119.3 -> 10) at 68 px = 6.6 -> 7
MARKER_R = {"PhC-C2DL-PSC": 3, "Fluo-N2DL-HeLa": 4, "DIC-C2DH-HeLa": 10, "Lead-BF": 7}
# library mask source: default <video>_ST/SEG/man_seg*.tif (labels == track ids for the CTC silver truth). Lead-BF:
# the SEG masks carry per-instance ids unique across frames (not track ids); its _GT/TRA/man_track*.tif are FULL masks
# (same foreground) labelled with the track ids and consistent with man_track.txt -> "tra"
LIB_MASK_SRC = {"Lead-BF": "tra"}
S4 = None  # 4-connectivity structure (ndi default)
S8 = np.ones((3, 3), bool)


def log(*a):
    print(*a, flush=True)


# ============================================================================ library
def build_library(dataset, video, root):
    """All tracks of the video as per-frame (cy, cx, bool crop); division templates; real statistics."""
    t0 = time.time()
    tr = np.loadtxt(f"{root}/{dataset}/{video}_GT/TRA/man_track.txt", dtype=np.int64, ndmin=2)
    tracks = {int(L): dict(B=int(B), E=int(E), P=int(P), fr={}) for L, B, E, P in tr}
    if LIB_MASK_SRC.get(dataset) == "tra":  # Lead-BF: full masks with track ids live in the TRA tifs
        segdir = f"{root}/{dataset}/{video}_GT/TRA"
        files = sorted(f for f in os.listdir(segdir) if f.endswith(".tif") and f.startswith("man_track"))
    else:
        segdir = f"{root}/{dataset}/{video}_ST/SEG"
        files = sorted(f for f in os.listdir(segdir) if f.endswith(".tif"))
    cells_per_frame, areas = [], []
    H = W = None
    for t, fn in enumerate(files):
        seg = tifffile.imread(os.path.join(segdir, fn)).astype(np.int32)
        H, W = seg.shape
        n = 0
        for lab, sl in enumerate(ndi.find_objects(seg), 1):
            if sl is None or lab not in tracks:
                continue
            m = seg[sl] == lab
            cc, k = ndi.label(m)
            if k > 1:  # keep the largest 4-connected component
                sizes = ndi.sum(m, cc, range(1, k + 1))
                m = cc == (1 + int(np.argmax(sizes)))
            ys, xs = np.nonzero(m)
            cy = sl[0].start + ys.mean()
            cx = sl[1].start + xs.mean()
            crop = np.ascontiguousarray(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
            tracks[lab]["fr"][t] = (float(cy), float(cx), crop)
            areas.append(int(crop.sum()))
            n += 1
        cells_per_frame.append(n)
    # contiguous runs of frames with a mask
    runs = []  # (track id, [frames])
    for L, tk in tracks.items():
        fs = sorted(tk["fr"])
        if not fs:
            continue
        cur = [fs[0]]
        for f in fs[1:]:
            if f == cur[-1] + 1:
                cur.append(f)
            else:
                runs.append((L, cur))
                cur = [f]
        runs.append((L, cur))
    children = {}
    for L, tk in tracks.items():
        if tk["P"] > 0:
            children.setdefault(tk["P"], []).append(L)
    mothers = {P: ch for P, ch in children.items() if len(ch) >= 2 and P in tracks}
    # division templates
    templates = []
    for M, ch in mothers.items():
        tm = tracks[M]
        E = tm["E"]
        if not all(f in tm["fr"] for f in (E - 2, E - 1, E)):
            continue
        ds = [d for d in ch if tracks[d]["B"] == E + 1 and all(f in tracks[d]["fr"] for f in (E + 1, E + 2, E + 3))]
        if len(ds) < 2:
            continue
        ay, ax = tm["fr"][E][0], tm["fr"][E][1]
        mother = [(tm["fr"][f][0] - ay, tm["fr"][f][1] - ax, tm["fr"][f][2]) for f in (E - 2, E - 1, E)]
        if E - 3 in tm["fr"]:
            pre_step = (tm["fr"][E - 2][0] - tm["fr"][E - 3][0], tm["fr"][E - 2][1] - tm["fr"][E - 3][1])
        else:
            pre_step = (0.0, 0.0)
        daughters = []
        for d in ds[:2]:
            td = tracks[d]["fr"]
            daughters.append([(td[f][0] - ay, td[f][1] - ax, td[f][2]) for f in (E + 1, E + 2, E + 3)])
        templates.append(dict(mother=M, daughters=ds[:2], m=mother, d=daughters, pre_step=pre_step))
    lib = dict(dataset=dataset, video=video, H=H, W=W, n_frames=len(files), tracks=tracks, runs=runs,
               mothers=mothers, templates=templates,
               real_cells_per_frame=float(np.mean(cells_per_frame)), real_area_median=float(np.median(areas)),
               real_area_mean=float(np.mean(areas)))
    log(f"  library: {len(tracks)} tracks, {len(runs)} runs, {len(templates)} division templates, "
        f"real cells/frame {lib['real_cells_per_frame']:.1f} (full {H}x{W}), median area {lib['real_area_median']:.0f}px "
        f"({time.time() - t0:.1f}s)")
    return lib


def build_library_multi(dataset, videos, root):
    """Merged library of several videos (comma-separated --video). One video -> build_library unchanged.
    Track ids of video i are offset by the sum of max ids of the previous videos (P/mothers/runs/templates
    remapped consistently); lib["provenance"][new_id] = (video, original id); templates keep their video.
    Pooled statistics: cells/frame weighted by frame count, area median/mean over all stored masks."""
    if len(videos) == 1:
        return build_library(dataset, videos[0], root)
    libs = [build_library(dataset, v, root) for v in videos]
    H, W = libs[0]["H"], libs[0]["W"]
    tracks, runs, mothers, templates, prov, areas = {}, [], {}, [], {}, []
    off, n_cells_total, n_frames_total, n_frames = 0, 0.0, 0, []
    for lb in libs:
        assert (lb["H"], lb["W"]) == (H, W), f"frame size mismatch {lb['video']}: {(lb['H'], lb['W'])} vs {(H, W)}"
        for L, tk in lb["tracks"].items():
            tracks[L + off] = dict(B=tk["B"], E=tk["E"], P=tk["P"] + off if tk["P"] > 0 else 0, fr=tk["fr"])
            prov[L + off] = (lb["video"], L)
            areas.extend(int(c.sum()) for _, _, c in tk["fr"].values())
        runs.extend((L + off, fr) for L, fr in lb["runs"])
        mothers.update({P + off: [c + off for c in ch] for P, ch in lb["mothers"].items()})
        templates.extend(dict(tp, mother=tp["mother"] + off, daughters=[d + off for d in tp["daughters"]],
                              video=lb["video"]) for tp in lb["templates"])
        n_frames.append(lb["n_frames"])
        n_cells_total += lb["real_cells_per_frame"] * lb["n_frames"]
        n_frames_total += lb["n_frames"]
        off += max(lb["tracks"]) if lb["tracks"] else 0
    lib = dict(dataset=dataset, video=",".join(videos), videos=list(videos), H=H, W=W, n_frames=n_frames_total,
               n_frames_per_video=n_frames, tracks=tracks, runs=runs, mothers=mothers, templates=templates,
               provenance=prov, real_cells_per_frame=n_cells_total / max(n_frames_total, 1),
               real_area_median=float(np.median(areas)), real_area_mean=float(np.mean(areas)))
    log(f"  merged library of {len(videos)} videos: {len(tracks)} tracks, {len(runs)} runs, {len(templates)} division "
        f"templates, real cells/frame {lib['real_cells_per_frame']:.1f} (full {H}x{W}), median area {lib['real_area_median']:.0f}px")
    return lib


# ============================================================================ transforms
def tf_crop(crop, k, flip):
    c = crop[::-1] if flip else crop
    return np.ascontiguousarray(np.rot90(c, k))


def tf_vec(dy, dx, k, flip):
    if flip:
        dy = -dy
    for _ in range(k):
        dy, dx = -dx, dy
    return dy, dx


class Sampler:
    """Segment / template sampling from the library with one D4 transform per unit."""

    def __init__(self, lib, rng):
        self.lib, self.rng = lib, rng
        tracks = lib["tracks"]
        runs = lib["runs"]
        self.any_runs = runs
        self.any_len = np.array([len(r[1]) for r in runs])
        self.mpre_runs = [(L, fr) for L, fr in runs if L in lib["mothers"] and fr[-1] == tracks[L]["E"]]
        self.mpre_len = np.array([len(r[1]) for r in self.mpre_runs])
        self.dpost_runs = [(L, fr) for L, fr in runs if tracks[L]["P"] > 0 and fr[0] == tracks[L]["B"]]
        self.dpost_len = np.array([len(r[1]) for r in self.dpost_runs])

    def _frames(self, L, frames, prev=None, nxt=None):
        """segment as list of (step_dy, step_dx, crop): step = centroid displacement from previous frame."""
        fr = self.lib["tracks"][L]["fr"]
        pos = [fr[f][:2] for f in frames]
        crops = [fr[f][2] for f in frames]
        if prev is not None:
            p0 = fr[prev][:2]
        else:
            p0 = pos[0]
        out = []
        for i in range(len(frames)):
            q = pos[i - 1] if i > 0 else p0
            out.append((pos[i][0] - q[0], pos[i][1] - q[1], crops[i]))
        return out

    def segment(self, L, kind, k, flip):
        """kind: 'any' (random offset, random time reversal), 'mother_pre' (ends 3 frames before a division),
        'daughter_post' (starts 3 frames after birth). Returns L items of (dy, dx, crop) transformed."""
        rng = self.rng
        if L <= 0:
            return []
        seg = None
        if kind == "mother_pre":
            idx = np.nonzero(self.mpre_len >= L + 3)[0]
            if len(idx):
                tid, fr = self.mpre_runs[int(rng.choice(idx))]
                sub = fr[len(fr) - 3 - L:len(fr) - 3]
                prev = fr[len(fr) - 4 - L] if len(fr) - 4 - L >= 0 else None
                seg = self._frames(tid, sub, prev=prev)
        elif kind == "daughter_post":
            idx = np.nonzero(self.dpost_len >= L + 3)[0]
            if len(idx):
                tid, fr = self.dpost_runs[int(rng.choice(idx))]
                seg = self._frames(tid, fr[3:3 + L], prev=fr[2])
        if seg is None:
            idx = np.nonzero(self.any_len >= L)[0]
            if not len(idx):  # should not happen for L <= 17
                idx = np.nonzero(self.any_len >= 1)[0]
            tid, fr = self.any_runs[int(rng.choice(idx))]
            if len(fr) < L:  # repeat the run (very short tracks only)
                fr = (fr * (L // len(fr) + 1))
                seg = self._frames(tid, fr[:1] * L)
            else:
                s = int(rng.integers(0, len(fr) - L + 1))
                if rng.random() < 0.5:  # time reversal
                    sub = fr[s:s + L][::-1]
                    nxt = fr[s + L] if s + L < len(fr) else None
                    seg = self._frames(tid, sub, prev=nxt)
                else:
                    seg = self._frames(tid, fr[s:s + L], prev=fr[s - 1] if s > 0 else None)
        return [(*tf_vec(dy, dx, k, flip), tf_crop(c, k, flip)) for dy, dx, c in seg]

    def template(self, k, flip):
        tp = self.lib["templates"][int(self.rng.integers(len(self.lib["templates"])))]
        T = lambda lst: [(*tf_vec(ry, rx, k, flip), tf_crop(c, k, flip)) for ry, rx, c in lst]
        ds = tp["d"] if self.rng.random() < 0.5 else tp["d"][::-1]
        return dict(m=T(tp["m"]), d=[T(ds[0]), T(ds[1])], pre_step=tf_vec(*tp["pre_step"], k, flip),
                    src=(tp["mother"], tp["daughters"]))


# ============================================================================ unit (one cell lineage)
def crop_cen(crop):
    ys, xs = np.nonzero(crop)
    return (float(ys.mean()), float(xs.mean()), int(len(ys)))


def finalize_unit(unit):
    for tk in unit["tracks"]:
        tk["cen"] = [crop_cen(c) for c in tk["crops"]]
    return unit


def build_unit(sampler, nf, div_frame=None):
    """A unit = one cell starting at frame 0 (optionally dividing at div_frame = daughters' birth frame).
    tracks: list of dict(B, pos=[(y,x) relative to unit origin], crops=[...], mark=[...], parent=idx or -1)."""
    rng = sampler.rng
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    tracks = []
    if div_frame is None:
        seg = sampler.segment(nf, "any", k, flip)
        pos, crops = [(0.0, 0.0)], [seg[0][2]]
        for dy, dx, c in seg[1:]:
            pos.append((pos[-1][0] + dy, pos[-1][1] + dx))
            crops.append(c)
        tracks.append(dict(B=0, pos=pos, crops=crops, mark=[0] * nf, parent=-1))
        return finalize_unit(dict(tracks=tracks, k=k, flip=flip, div=None, src=None))
    td = div_frame
    pre = sampler.segment(td - 3, "mother_pre", k, flip)
    tp = sampler.template(k, flip)
    pos, crops = [], []
    if pre:
        pos, crops = [(0.0, 0.0)], [pre[0][2]]
        for dy, dx, c in pre[1:]:
            pos.append((pos[-1][0] + dy, pos[-1][1] + dx))
            crops.append(c)
        anchor = (pos[-1][0] + tp["pre_step"][0] - tp["m"][0][0], pos[-1][1] + tp["pre_step"][1] - tp["m"][0][1])
    else:
        anchor = (-tp["m"][0][0], -tp["m"][0][1])
    for ry, rx, c in tp["m"]:
        pos.append((anchor[0] + ry, anchor[1] + rx))
        crops.append(c)
    mark = [0] * td
    mark[td - 3:td] = [255, 255, 255]
    tracks.append(dict(B=0, pos=pos, crops=crops, mark=mark, parent=-1))
    for di in range(2):
        dpos, dcrops = [], []
        for ry, rx, c in tp["d"][di]:
            dpos.append((anchor[0] + ry, anchor[1] + rx))
            dcrops.append(c)
        L = nf - td - 3
        if L > 0:
            seg = sampler.segment(L, "daughter_post", k, flip)
            for dy, dx, c in seg:
                dpos.append((dpos[-1][0] + dy, dpos[-1][1] + dx))
                dcrops.append(c)
        dpos, dcrops = dpos[:nf - td], dcrops[:nf - td]
        mark = [0] * len(dpos)
        mark[0] = 128
        tracks.append(dict(B=td, pos=dpos, crops=dcrops, mark=mark, parent=0))
    return finalize_unit(dict(tracks=tracks, k=k, flip=flip, div=td, src=tp["src"]))


# ============================================================================ canvas / placement
class Canvas:
    def __init__(self, nf, H, W):
        self.nf, self.H, self.W = nf, H, W
        self.lab = np.zeros((nf, H, W), np.uint16)
        self.cent = [dict() for _ in range(nf)]  # frame -> id -> (y, x) intended centroid
        self.area_int = [dict() for _ in range(nf)]  # intended (unclipped-by-others) area
        self.next_id = 1
        self.resolved_px = 0
        self.lost_px = 0

    def placements(self, unit, oy, ox, strict):
        """Resolve a unit at origin (oy, ox) into per-track per-frame pastes.
        Returns list of tracks: (B, E, parent, [(t, y, x, crop_clipped, sl)]) or None if invalid.
        strict: every frame must be fully inside the canvas (dividing units); else tracks may leave (track ends)."""
        H, W = self.H, self.W
        out = []
        for ti, tk in enumerate(unit["tracks"]):
            items = []
            for i, ((py, px), crop, (cy, cx, area)) in enumerate(zip(tk["pos"], tk["crops"], tk["cen"])):
                t = tk["B"] + i
                if t >= self.nf:
                    break
                h, w = crop.shape
                y = oy + py
                x = ox + px
                ty = int(round(y - cy))
                tx = int(round(x - cx))
                y0, x0, y1, x1 = max(ty, 0), max(tx, 0), min(ty + h, H), min(tx + w, W)
                full = ty >= 0 and tx >= 0 and ty + h <= H and tx + w <= W
                if y1 <= y0 or x1 <= x0:
                    sub = None
                elif full:
                    sub = crop
                else:
                    sub = crop[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
                    if sub.sum() < 0.3 * area or sub.sum() < 4:
                        sub = None
                if sub is None or (strict and not full):
                    if strict or i == 0:
                        return None
                    break  # track leaves the field of view: ends at previous frame
                items.append((t, y, x, sub, (slice(y0, y1), slice(x0, x1))))
            if not items:
                return None
            out.append((tk["B"], tk["B"] + len(items) - 1, tk["parent"], items, tk["mark"][:len(items)]))
        # a mother must reach her division frame
        if unit["div"] is not None and out[0][1] != unit["div"] - 1:
            return None
        return out

    def overlap(self, pl):
        n = 0
        for B, E, par, items, mark in pl:
            for t, y, x, sub, sl in items:
                n += int(np.count_nonzero(self.lab[t][sl][sub]))
        return n

    def paint(self, pl, min_keep=0.6):
        """Paint the unit; contested pixels go to the nearer centroid; one 4-connected component per id;
        undo and return None if any cell (new or existing) keeps < min_keep of its intended area."""
        undo = []
        new_ids = []
        ok = True
        id_map = {}
        marks = {}
        for ti, (B, E, par, items, mark) in enumerate(pl):
            nid = self.next_id
            self.next_id += 1
            new_ids.append(nid)
            id_map[ti] = nid
            pid = id_map[par] if par >= 0 else 0
            marks[nid] = (B, E, pid, mark)
            for (t, y, x, sub, sl), mk in zip(items, mark):
                region = self.lab[t][sl]
                undo.append((t, sl, region.copy()))
                occ = region != 0
                take = sub & ~occ
                contested = sub & occ
                affected = []
                if contested.any():
                    yy, xx = np.nonzero(contested)
                    owners = region[yy, xx].astype(np.int64)
                    gy, gx = yy + sl[0].start, xx + sl[1].start
                    d_new = (gy - y) ** 2 + (gx - x) ** 2
                    cen = self.cent[t]
                    oc = np.array([cen[int(o)] for o in owners], np.float64).reshape(-1, 2)
                    d_own = (gy - oc[:, 0]) ** 2 + (gx - oc[:, 1]) ** 2
                    win = d_new < d_own
                    take[yy[win], xx[win]] = True
                    affected = np.unique(owners[win]).tolist()
                    self.resolved_px += int(contested.sum())
                region[take] = nid
                # one component for the new id (within its own bbox)
                m = region == nid
                cc, k = ndi.label(m)
                if k > 1:
                    sizes = ndi.sum(m, cc, range(1, k + 1))
                    keep = 1 + int(np.argmax(sizes))
                    region[m & (cc != keep)] = 0
                    self.lost_px += int(sizes.sum() - sizes.max())
                a_new = int((region == nid).sum())
                a_int = int(sub.sum())
                self.cent[t][nid] = (float(y), float(x))
                self.area_int[t][nid] = a_int
                if a_new < min_keep * a_int:
                    ok = False
                    break
                # one component + area for every affected existing id (search a window around the paste)
                for o in affected:
                    o = int(o)
                    if not self._fix_id(t, o, sl, undo, min_keep):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            for t, sl, saved in reversed(undo):
                self.lab[t][sl] = saved
            for t in range(self.nf):
                for nid in new_ids:
                    self.cent[t].pop(nid, None)
                    self.area_int[t].pop(nid, None)
            self.next_id = new_ids[0]
            return None
        return marks

    def _fix_id(self, t, o, sl, undo, min_keep):
        """Keep the largest component of existing id o in frame t (its mask lies within a window around sl)."""
        H, W = self.H, self.W
        r = 4 * int(math.sqrt(self.area_int[t][o])) + 4
        cy, cx = self.cent[t][o]
        wy0, wy1 = max(int(cy) - r, 0), min(int(cy) + r, H)
        wx0, wx1 = max(int(cx) - r, 0), min(int(cx) + r, W)
        wy0, wy1 = min(wy0, sl[0].start), max(wy1, sl[0].stop)
        wx0, wx1 = min(wx0, sl[1].start), max(wx1, sl[1].stop)
        wsl = (slice(wy0, wy1), slice(wx0, wx1))
        region = self.lab[t][wsl]
        m = region == o
        a = int(m.sum())
        if a < min_keep * self.area_int[t][o]:
            return False
        cc, k = ndi.label(m)
        if k > 1:
            undo.append((t, wsl, region.copy()))
            sizes = ndi.sum(m, cc, range(1, k + 1))
            keep = 1 + int(np.argmax(sizes))
            region[m & (cc != keep)] = 0
            self.lost_px += int(sizes.sum() - sizes.max())
            if sizes.max() < min_keep * self.area_int[t][o]:
                return False
        return True


def place_unit(canvas, sampler, unit, rng, contact, n_pos=12, margin=2):
    """Choose an origin for the unit (contact: adjacent to an existing frame-0 cell), minimising overlap."""
    H, W = canvas.H, canvas.W
    c0 = unit["tracks"][0]["crops"][0]
    h0, w0 = c0.shape
    r0 = math.sqrt(c0.sum() / math.pi)
    # extent of the unit relative to its origin (to keep strict units inside)
    strict = unit["div"] is not None
    ids0 = list(canvas.cent[0].keys())
    cands = []
    for _ in range(n_pos):
        if contact and ids0:
            j = ids0[int(rng.integers(len(ids0)))]
            cy, cx = canvas.cent[0][j]
            rj = math.sqrt(canvas.area_int[0][j] / math.pi)
            th = rng.random() * 2 * math.pi
            d = r0 + rj + rng.integers(-1, 2)
            oy, ox = cy + d * math.sin(th), cx + d * math.cos(th)
        else:
            oy = rng.uniform(h0 / 2 + margin, H - h0 / 2 - margin)
            ox = rng.uniform(w0 / 2 + margin, W - w0 / 2 - margin)
        pl = canvas.placements(unit, oy, ox, strict)
        if pl is None:
            continue
        ov = canvas.overlap(pl)
        cands.append((ov, oy, ox, pl))
        if ov == 0 and not contact:
            break
        if contact and ov <= 0.15 * c0.sum():
            break
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    ov, oy, ox, pl = cands[0]
    marks = canvas.paint(pl)
    if marks is None:
        return None
    return dict(marks=marks, origin=(float(oy), float(ox)), overlap=ov)


# ============================================================================ outputs
def cond_image(lab, markval, version=1):
    """shared renderer (cond_render.py). v1: R contour, G EDT, B = event marker value per id (LUT); v2: see cond_render."""
    return render_cond(lab, None, 0, version, B=markval[lab])


def markers_image(lab, radius):
    """CTC TRA markers: disc of `radius` at each instance centroid, restricted to the instance mask."""
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
        # one 8-connected marker blob per id (disc clipped to a concave mask can split): keep the component
        # that contains (cy, cx), which is guaranteed to lie inside the mask  [verifier fix]
        cc, k = ndi.label(m, structure=S8)
        if k > 1:
            m = cc == cc[cy - y0, cx - x0]
        win[m] = lid
        if not (out == lid).any():
            out[cy, cx] = lid
    return out


def contact_pairs(lab):
    """number of distinct touching instance pairs (8-connectivity)."""
    pairs = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lab[max(dy, 0):lab.shape[0] - max(-dy, 0), max(dx, 0):lab.shape[1] - max(-dx, 0)]
        b = lab[max(-dy, 0):lab.shape[0] - max(dy, 0), max(-dx, 0):lab.shape[1] - max(dx, 0)]
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            p = np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1)
            pairs.update(map(tuple, np.unique(p, axis=0).tolist()))
    return len(pairs)


def load_ref(dataset, video, root, rng, H, W):
    """random real frame of video 02 as 8-bit gray x3 (fixed per-dataset intensity mapping of the real clips).
    `video` may be a comma-separated list: one listed video is drawn first (recorded as ref "video")."""
    videos = video.split(",")
    if len(videos) > 1:
        video = videos[int(rng.integers(len(videos)))]
    img_dir = f"{root}/{dataset}/{video}"
    nfr = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    t = int(rng.integers(nfr))
    im = tifffile.imread(f"{img_dir}/t{t:03d}.tif")
    crops = REF_CROPS[dataset]
    crop = crops[int(rng.integers(len(crops)))]
    if crop is not None:
        y0, x0, h, w, name = crop
        im = im[y0:y0 + h, x0:x0 + w]
        cinfo = {"y0": y0, "x0": x0, "h": h, "w": w, "name": name}
    else:
        cinfo = None
    mp = f"{N}/data/vace_clips/{dataset}/intensity_mapping.json"
    if os.path.exists(mp):
        with open(mp) as f:
            mj = json.load(f)
        lo, hi = float(mj.get("lo", np.percentile(im, 1))), float(mj.get("hi", np.percentile(im, 99.9)))
        src = mp
    else:
        lo, hi = float(np.percentile(im, 1.0)), float(np.percentile(im, 99.9))
        src = "frame percentiles 1/99.9"
    x = (im.astype(np.float32) - lo) / max(hi - lo, 1.0)
    im8 = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    assert im8.shape == (H, W), (im8.shape, (H, W))
    info = {"frame": t, "crop": cinfo, "lo": lo, "hi": hi, "mapping": src}
    if len(videos) > 1:
        info["video"] = video
    return np.stack([im8] * 3, -1), info


# ============================================================================ clip generation
def make_clip(lib, args, clip_idx, out_dir, ctc_dir, H, W):
    t0 = time.time()
    seed = args.seed * 100000 + clip_idx
    rng = np.random.default_rng(seed)
    sampler = Sampler(lib, rng)
    nf = args.num_frames
    canvas = Canvas(nf, H, W)
    a, b = args.n_cells_range
    n_cells = max(1, int(round(rng.integers(a, b + 1) * args.density_scale)))
    da, db = args.divisions_per_clip_range
    n_div = int(rng.integers(da, db + 1))
    n_div = min(n_div, n_cells)
    div_frames = [int(rng.integers(3, nf - 1)) for _ in range(n_div)]  # daughters born at td in [3, nf-2]
    plan = [("div", td) for td in div_frames] + [("plain", None)] * (n_cells - n_div)
    units_meta = []
    lineage = {}  # id -> (B, E, P)
    marks = {}  # id -> list of per-frame mark values from B
    skipped = {"div": 0, "plain": 0}
    n_contact = 0
    for kind, td in plan:
        placed = None
        contact = rng.random() < args.contact_prob
        for attempt in range(6):
            unit = build_unit(sampler, nf, td)
            placed = place_unit(canvas, sampler, unit, rng, contact and attempt < 4)
            if placed is not None:
                break
        if placed is None:
            skipped[kind] += 1
            continue
        n_contact += int(contact)
        for nid, (B, E, P, mk) in placed["marks"].items():
            lineage[nid] = (B, E, P)
            marks[nid] = mk
        units_meta.append(dict(kind=kind, div_frame=td, ids=sorted(placed["marks"]), contact=bool(contact),
                               origin=placed["origin"], overlap_at_placement=placed["overlap"],
                               transform=(unit["k"], unit["flip"]), template_src=unit["src"]))
    # ---- verification from the final label stack
    lab = canvas.lab
    maxid = canvas.next_id
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
            k8 = ndi.label(m, structure=S8)[1]
            if k4 != 1:
                one_cc4 += 1
                cc_list.append((lid, t, int(k4)))
            if k8 != 1:
                one_cc8 += 1
    lineage_bad = 0
    for lid, (B, E, P) in lineage.items():
        fr = present.get(lid, [])
        if fr != list(range(B, E + 1)):
            lineage_bad += 1
        if P > 0 and lineage[P][1] + 1 != B:
            lineage_bad += 1
    if set(present) != set(lineage):
        lineage_bad += 1
    cells = [int(np.sum([1 for lid in present if t in present[lid]])) for t in range(nf)]
    cpairs = [contact_pairs(lab[t]) for t in range(nf)]
    n_div_done = sum(1 for u in units_meta if u["kind"] == "div")
    # ---- write
    cid = f"clip_{clip_idx:03d}"
    cdir = os.path.join(out_dir, cid)
    cond_dir = cond_dirname(args.cond_version)
    for sub in ("labels", cond_dir):
        os.makedirs(os.path.join(cdir, sub), exist_ok=True)
    tra_dir = os.path.join(ctc_dir, f"{cid}_GT", "TRA")
    seg_dir = os.path.join(ctc_dir, f"{cid}_ST", "SEG")
    os.makedirs(tra_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    markval_t = np.zeros((nf, maxid), np.uint8)
    for lid, mk in marks.items():
        B = lineage[lid][0]
        for i, v in enumerate(mk):
            markval_t[B + i, lid] = v
    rad = MARKER_R[args.dataset]
    n_mother_px = n_daughter_px = 0
    for t in range(nf):
        L = lab[t]
        cond = cond_image(L, markval_t[t], args.cond_version)
        n_mother_px += int((cond[..., 2] == 255).sum())
        n_daughter_px += int((cond[..., 2] == 128).sum())
        cv2.imwrite(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), L, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(cdir, cond_dir, f"frame_{t:03d}.png"), cond[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tifffile.imwrite(os.path.join(seg_dir, f"man_seg{t:03d}.tif"), L, compression="zlib")
        tifffile.imwrite(os.path.join(tra_dir, f"man_track{t:03d}.tif"), markers_image(L, rad), compression="zlib")
    lines = [f"{lid} {B} {E} {P}\n" for lid, (B, E, P) in sorted(lineage.items())]
    with open(os.path.join(cdir, "lineage.txt"), "w") as f:
        f.writelines(lines)
    with open(os.path.join(tra_dir, "man_track.txt"), "w") as f:
        f.writelines(lines)
    ref, ref_info = load_ref(args.dataset, args.video, args.root, rng, H, W)
    cv2.imwrite(os.path.join(cdir, "ref.png"), ref[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    meta = dict(dataset=args.dataset, video=args.video, clip=cid, seed=seed, num_frames=nf, height=H, width=W,
                synthetic=True, generator="lineage_sim.py", cond_version=int(args.cond_version), cond_dir=cond_dir,
                params=dict(n_cells_range=list(args.n_cells_range), divisions_per_clip_range=list(args.divisions_per_clip_range),
                            contact_prob=args.contact_prob, density_scale=args.density_scale),
                n_cells_planned=n_cells, n_div_planned=n_div, n_div=n_div_done, division_frames=[u["div_frame"] for u in units_meta if u["kind"] == "div"],
                n_ids=len(lineage), cells_per_frame=cells, mean_cells_per_frame=float(np.mean(cells)),
                contact_pairs_per_frame=cpairs, mean_contact_pairs_per_frame=float(np.mean(cpairs)),
                contact_placed_units=n_contact, skipped_units=skipped,
                resolved_overlap_px=canvas.resolved_px, lost_fragment_px=canvas.lost_px,
                overlap_violations=0, violations_one_cc=one_cc4, violations_one_cc_8conn=one_cc8,
                violations_one_cc_list=cc_list[:50], lineage_violations=lineage_bad,
                mother_marked_px=n_mother_px, daughter_marked_px=n_daughter_px,
                ref=ref_info, units=units_meta, seconds=time.time() - t0)
    with open(os.path.join(cdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ============================================================================ montage
def colorize(lab, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    lut = rng.integers(60, 256, size=(int(lab.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    return lut[lab]


def montage(out_dir, metas, fig_path, nf, scale=0.5):
    frames = [0, nf // 2, nf - 1]
    rows = []
    for m in metas[:3]:
        cdir = os.path.join(out_dir, m["clip"])
        conds = [cv2.imread(os.path.join(cdir, m.get("cond_dir", "cond"), f"frame_{t:03d}.png"), -1)[..., ::-1] for t in frames]
        labs = [colorize(cv2.imread(os.path.join(cdir, "labels", f"frame_{t:03d}.png"), -1)) for t in frames]
        for row in (conds, labs):
            row = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for im in row]
            rows.append(np.concatenate(row, 1))
    img = np.concatenate(rows, 0)
    # labels
    y = 0
    for i, m in enumerate(metas[:3]):
        for j, name in enumerate(("cond", "labels")):
            cv2.putText(img, f"{m['clip']} {name}  cells={m['mean_cells_per_frame']:.0f} div={m['n_div']}", (6, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += rows[2 * i + j].shape[0]
    for j, t in enumerate(frames):
        cv2.putText(img, f"frame {t}", (j * rows[0].shape[1] // 3 + rows[0].shape[1] // 3 - 80, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    cv2.imwrite(fig_path, np.ascontiguousarray(img[..., ::-1]))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CANVAS))
    ap.add_argument("--video", default="02")
    ap.add_argument("--num_clips", type=int, default=40)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--n_cells_range", default=None, help="a,b cells at frame 0 (default: 0.8..1.2 x real cells/frame, "
                                                            "scaled to the canvas area); multiplied by density_scale")
    ap.add_argument("--divisions_per_clip_range", default="1,3")
    ap.add_argument("--contact_prob", type=float, default=0.3)
    ap.add_argument("--density_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=f"{N}/data/ctc")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--cache", default=None, help="library pickle (default <out>/../_library_<video>.pkl)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cond_version", type=int, default=1, choices=[1, 2],
                    help="condition encoding (cond_render.py): 1 = original -> cond/, 2 -> cond_v2/")
    args = ap.parse_args()
    videos = args.video.split(",")  # comma-separated list allowed (single video: unchanged behaviour)
    if "01" in videos:
        sys.exit("REFUSING: video 01 is the held-out evaluation video and must never be used to build the library/refs.")
    args.divisions_per_clip_range = tuple(int(v) for v in args.divisions_per_clip_range.split(","))
    H, W = CANVAS[args.dataset]
    out_dir = args.out.rstrip("/")
    ctc_dir = out_dir + "_ctc"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ctc_dir, exist_ok=True)
    cache_name = "+".join(videos)
    if len(cache_name) > 200:  # many videos: the joined name would exceed the 255-byte file name limit
        import hashlib
        cache_name = f"{videos[0]}+...+{videos[-1]}_n{len(videos)}_{hashlib.md5(cache_name.encode()).hexdigest()[:8]}"
    cache = args.cache or os.path.join(os.path.dirname(out_dir), f"_library_{cache_name}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            lib = pickle.load(f)
        log(f"  library loaded from {cache}: {len(lib['tracks'])} tracks, {len(lib['templates'])} templates")
    else:
        lib = build_library_multi(args.dataset, videos, args.root)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(lib, f, protocol=4)
    area_ratio = (H * W) / float(lib["H"] * lib["W"])
    real_mean = lib["real_cells_per_frame"] * area_ratio
    if args.n_cells_range:
        args.n_cells_range = tuple(int(v) for v in args.n_cells_range.split(","))
    else:
        args.n_cells_range = (int(round(0.8 * real_mean)), int(round(1.2 * real_mean)))
    log(f"[{args.dataset}] canvas {H}x{W}, real cells/frame (canvas-scaled) {real_mean:.1f}, n_cells_range {args.n_cells_range} "
        f"x density_scale {args.density_scale}, divisions {args.divisions_per_clip_range}, contact_prob {args.contact_prob}, "
        f"seed {args.seed}, {args.num_clips} clips -> {out_dir}")
    t0 = time.time()
    jobs = [(clip_idx, out_dir, ctc_dir, H, W) for clip_idx in range(args.num_clips)]
    if args.workers > 1:
        from multiprocessing import Pool
        global _LIB, _ARGS
        _LIB, _ARGS = lib, args
        with Pool(args.workers) as pool:
            metas = pool.map(_job, jobs)
    else:
        metas = [make_clip(lib, args, *j) for j in jobs]
    wall = time.time() - t0
    for m in metas:
        log(f"  {m['clip']}: cells/frame {m['mean_cells_per_frame']:.1f} div {m['n_div']} ({m['division_frames']}) "
            f"contact pairs/frame {m['mean_contact_pairs_per_frame']:.1f} resolved_px {m['resolved_overlap_px']} "
            f"one_cc_viol {m['violations_one_cc']} lineage_viol {m['lineage_violations']} skipped {m['skipped_units']} "
            f"{m['seconds']:.1f}s")
    summary = dict(dataset=args.dataset, video=args.video, out=out_dir, ctc=ctc_dir, num_clips=len(metas),
                   params=dict(n_cells_range=list(args.n_cells_range), divisions_per_clip_range=list(args.divisions_per_clip_range),
                               contact_prob=args.contact_prob, density_scale=args.density_scale, seed=args.seed,
                               num_frames=args.num_frames),
                   real_cells_per_frame_canvas=real_mean,
                   mean_cells_per_frame=float(np.mean([m["mean_cells_per_frame"] for m in metas])),
                   divisions_total=int(sum(m["n_div"] for m in metas)),
                   divisions_per_clip_mean=float(np.mean([m["n_div"] for m in metas])),
                   divisions_planned_total=int(sum(m["n_div_planned"] for m in metas)),
                   mean_contact_pairs_per_frame=float(np.mean([m["mean_contact_pairs_per_frame"] for m in metas])),
                   overlap_violations=int(sum(m["overlap_violations"] for m in metas)),
                   one_cc_violations=int(sum(m["violations_one_cc"] for m in metas)),
                   one_cc_violations_8conn=int(sum(m["violations_one_cc_8conn"] for m in metas)),
                   lineage_violations=int(sum(m["lineage_violations"] for m in metas)),
                   resolved_overlap_px_total=int(sum(m["resolved_overlap_px"] for m in metas)),
                   skipped_units=dict(div=int(sum(m["skipped_units"]["div"] for m in metas)), plain=int(sum(m["skipped_units"]["plain"] for m in metas))),
                   seconds_per_clip_single=float(np.mean([m["seconds"] for m in metas])),
                   wall_seconds=wall, workers=args.workers, wall_seconds_per_clip=wall / max(len(metas), 1))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    txt = "\n".join(f"{k}: {v}" for k, v in summary.items())
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    log(txt)
    if args.fig:
        montage(out_dir, metas, args.fig, args.num_frames)
        log(f"  montage -> {args.fig}")


def _job(j):
    return make_clip(_LIB, _ARGS, *j)


if __name__ == "__main__":
    main()
