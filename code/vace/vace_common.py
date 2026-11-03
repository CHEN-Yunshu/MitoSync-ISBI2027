"""Shared helpers for the VACE cell-video pipeline.

Control-image convention (RGB, uint8, lossless PNG):
  R = 1-px instance contours (255 on the contour, else 0)
  G = per-instance normalized distance transform (0..255, 255 at the cell centre)
  B = 0 (reserved)
Reference image = a real microscopy frame converted to RGB 8-bit.
"""
import os
import glob
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


def to_uint8(img):
    """Convert a 2D microscopy frame (uint8/uint16/float) to uint8 using 1-99.8 percentile stretch for >8-bit."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, (0.5, 99.8))
    out = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return out.astype(np.uint8)


def gray_to_rgb_pil(img2d):
    g = to_uint8(img2d)
    return Image.fromarray(np.stack([g, g, g], axis=-1), mode="RGB")


def render_control(mask):
    """Instance label mask (H,W int) -> RGB uint8 control image (contours, normalized DT, 0)."""
    mask = np.asarray(mask)
    h, w = mask.shape
    ctrl = np.zeros((h, w, 3), dtype=np.uint8)
    if mask.max() == 0:
        return ctrl
    # 1-px contour: inner boundary of each instance (mode='inner' keeps it inside the object, 1 px wide)
    bnd = find_boundaries(mask, mode="inner", connectivity=1)
    ctrl[..., 0] = (bnd * 255).astype(np.uint8)
    # per-instance normalized distance transform
    dt = np.zeros((h, w), dtype=np.float32)
    objs = ndi.find_objects(mask)
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = mask[sl] == lab
        # pad so the distance transform sees the object border
        padded = np.pad(sub, 1)
        d = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
        m = d.max()
        if m > 0:
            d = d / m
        dt[sl][sub] = d[sub]
    ctrl[..., 1] = np.round(dt * 255).astype(np.uint8)
    return ctrl


def ctc_paths(ctc_root, dataset, video):
    img_dir = os.path.join(ctc_root, dataset, video)
    seg_dir = os.path.join(ctc_root, dataset, f"{video}_ST", "SEG")
    return img_dir, seg_dir


def load_ctc_frame(img_dir, t):
    return tifffile.imread(os.path.join(img_dir, f"t{t:03d}.tif"))


def load_ctc_seg(seg_dir, t):
    return tifffile.imread(os.path.join(seg_dir, f"man_seg{t:03d}.tif"))


def list_pngs(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNG frames in {folder}")
    return files


def montage(rows, labels=None, pad=4):
    """rows: list of lists of PIL images (same size) -> one PIL image (grid)."""
    from PIL import ImageDraw
    w, h = rows[0][0].size
    ncol = max(len(r) for r in rows)
    out = Image.new("RGB", (ncol * (w + pad) + pad, len(rows) * (h + pad) + pad), (30, 30, 30))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            out.paste(im.convert("RGB"), (pad + j * (w + pad), pad + i * (h + pad)))
    if labels:
        d = ImageDraw.Draw(out)
        for i, lab in enumerate(labels):
            d.text((pad + 4, pad + i * (h + pad) + 4), lab, fill=(255, 255, 0))
    return out
