#!/usr/bin/env python
"""
cond_render.py -- the ONE shared renderer of VACE condition images for cell-lineage label videos.
Used by make_clips.py (real clips), lineage_sim.py (simulated clips) and regen_cond_v2.py (re-encoding
existing corpora). Inference must use the SAME version the LoRA was trained with (cond/ = v1, cond_v2/ = v2).

cond_image(labels, lineage, t, version=1, mothers=None, daughters=None, B=None) -> uint8 (H, W, 3) RGB

  labels   uint16/int (H, W) instance labels (0 = background), one instance per track id
  lineage  dict id -> (B, E, P) in the SAME frame indexing as t (clip-local or global, caller's choice)
  t        frame index of `labels` in that indexing
  version  1 | 2 (see below)
  mothers / daughters  optional precomputed {mother_id: last_frame} / {daughter_id: birth_frame}
           (make_clips.py passes the GLOBAL ones so that the v1 output is byte-identical to the original encoder;
           when None they are derived from `lineage`: a mother = parent of >= 2 children)
  B        optional explicit uint8 (H, W) event channel (lineage_sim.py passes its per-id mark LUT values);
           overrides the lineage-derived marks (version 1 only; ignored for version 2)

version 1 (original encoding, cond/):
  R = 1-px inner contour of every instance (skimage find_boundaries mode='inner', connectivity=1) -> 255
  G = per-instance normalised EDT (0..255, 255 at the instance centre; EDT of the padded instance mask)
  B = event marks: mother pixels in her last 3 frames (E-2..E) -> 255, daughter pixels in the birth frame -> 128
version 2 (cond_v2/):
  R = v1 contour (255) PLUS the separation line: every pixel (foreground OR background) whose 3x3 neighbourhood
      contains >= 2 distinct instance ids, i.e. pixels within 1 px (8-neighbourhood) of a pixel of ANOTHER
      instance -> 255. Across a direct contact the R line is 2 px wide (1 px on each side, plus the background
      pixels capping the ends of the interface); across a 1-px background gap the gap pixel is added, so with
      the two inner contours the R line is 3 px wide (contour | gap | contour).
  G = v1 EDT, but forced to 0 on the separation line (the valley between touching instances is unmistakable)
  B = 0 everywhere (the v1 event marks were rendered as brightness by the v1 LoRAs -> removed)
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

VERSIONS = (1, 2)
COND_DIRNAME = {1: "cond", 2: "cond_v2"}


def cond_dirname(version):
    return COND_DIRNAME[int(version)]


def division_events(lineage):
    """mothers {mid: last_frame} (parents with >= 2 children present in `lineage`), daughters {did: birth_frame}."""
    children = {}
    for L, (B, E, P) in lineage.items():
        if P > 0:
            children.setdefault(P, []).append(L)
    mothers, daughters = {}, {}
    for P, ch in children.items():
        if len(ch) >= 2 and P in lineage:
            mothers[P] = lineage[P][1]
            for d in ch:
                daughters[d] = lineage[d][0]
    return mothers, daughters


def contour_channel(lab):
    return find_boundaries(lab, mode="inner", connectivity=1).astype(np.uint8) * 255


def edt_channel(lab, objs=None):
    h, w = lab.shape
    G = np.zeros((h, w), np.uint8)
    if objs is None:
        objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        sub = lab[sl] == i + 1
        if not sub.any():
            continue
        edt = ndi.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1]
        m = edt.max()
        g = np.round(edt / m * 255.0).astype(np.uint8) if m > 0 else np.zeros_like(sub, np.uint8)
        G[sl][sub] = g[sub]
    return G


def event_channel(lab, t, mothers, daughters, objs=None):
    h, w = lab.shape
    B = np.zeros((h, w), np.uint8)
    if objs is None:
        objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        lid = i + 1
        sub = lab[sl] == lid
        if not sub.any():
            continue
        if lid in mothers and mothers[lid] - 2 <= t <= mothers[lid]:
            B[sl][sub] = 255
        elif lid in daughters and daughters[lid] == t:
            B[sl][sub] = 128
    return B


def separation_mask(lab):
    """bool (H, W): pixels whose 3x3 neighbourhood (incl. itself) contains >= 2 distinct instance ids."""
    # int32 with a moderate sentinel: scipy's filters round an int64 sentinel of 2**63-1 through float64 and wrap it
    # negative, which marked every pixel next to any instance (caught by the self-test below)
    l32 = np.asarray(lab).astype(np.int32)
    big = np.int32(1 << 30)
    mx = ndi.maximum_filter(l32, size=3, mode="constant", cval=0)
    mn = ndi.minimum_filter(np.where(l32 > 0, l32, big), size=3, mode="constant", cval=big)
    return (mx > 0) & (mn < big) & (mx != mn)


def cond_image(labels, lineage, t, version=1, mothers=None, daughters=None, B=None):
    version = int(version)
    if version not in VERSIONS:
        raise ValueError(f"unknown cond version {version}")
    lab = np.asarray(labels)
    h, w = lab.shape
    cond = np.zeros((h, w, 3), np.uint8)
    if not (lab > 0).any():
        return cond
    objs = ndi.find_objects(lab)
    R = contour_channel(lab)
    G = edt_channel(lab, objs)
    if version == 1:
        if B is None:
            if mothers is None or daughters is None:
                mothers, daughters = division_events(lineage or {})
            B = event_channel(lab, t, mothers, daughters, objs)
        cond[..., 0] = R
        cond[..., 1] = G
        cond[..., 2] = np.asarray(B, np.uint8)
        return cond
    sep = separation_mask(lab)
    R = R.copy()
    R[sep] = 255
    G = G.copy()
    G[sep] = 0
    cond[..., 0] = R
    cond[..., 1] = G
    return cond


def load_lineage(path):
    """lineage.txt / man_track.txt (L B E P per line) -> dict id -> (B, E, P)."""
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                out[int(p[0])] = (int(p[1]), int(p[2]), int(p[3]))
    return out


def _selftest():
    lab = np.zeros((9, 12), np.uint16)
    lab[2:7, 1:5] = 1          # instance 1
    lab[2:7, 5:8] = 2          # instance 2 touching 1 at columns 4|5
    lab[2:7, 9:12] = 3         # instance 3 with a 1-px background gap (column 8) to instance 2
    sep = separation_mask(lab)
    # touching interface 4|5: 1 px on each side (rows 2..6) + the background pixels capping the interface ends (rows 1, 7)
    assert sep[2:7, 4].all() and sep[2:7, 5].all() and sep[:, 4].sum() == 7 and sep[:, 5].sum() == 7, "touching: 2 px wide"
    # 1-px background gap 7|8|9: only the gap pixel is within 1 px of BOTH instances
    assert sep[2:7, 8].all() and sep[:, 8].sum() == 7 and not sep[:, 7].any() and not sep[:, 9].any(), "gap: gap pixel only"
    assert not sep[:, [0, 1, 2, 3, 6, 10, 11]].any(), "no separation away from the interfaces"
    assert sep[0].sum() == 0 and sep[8].sum() == 0 and sep[1].sum() == 3 and sep[7].sum() == 3, "end caps only"
    c2 = cond_image(lab, {}, 0, 2)
    assert c2[..., 2].max() == 0 and (c2[2:7, 4:6, 0] == 255).all() and (c2[2:7, 4:6, 1] == 0).all(), "contact: 2-px R line, G=0"
    assert (c2[2:7, 7:10, 0] == 255).all() and (c2[2:7, 8, 1] == 0).all(), "gap: 3-px R line (contour, gap, contour)"
    assert np.array_equal(c2[2:7, 1:4], cond_image(lab, {}, 0, 1)[2:7, 1:4]), "away from the interface v2 == v1 (R, G)"
    c1 = cond_image(lab, {1: (0, 3, 0), 2: (4, 8, 1), 3: (4, 8, 1)}, 2, 1)
    assert (c1[2:7, 1:5, 2] == 255).all() and (c1[2:7, 5:8, 2] == 0).all()
    return True


if __name__ == "__main__":
    print("cond_render self-test:", _selftest())
