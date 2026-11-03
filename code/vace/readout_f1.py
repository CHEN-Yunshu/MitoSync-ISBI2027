#!/usr/bin/env python
"""Cellpose-SAM read-out gate for generated clips (run with the cellres_cp4_blackwell env).

Segments the frames of a clip with Cellpose-SAM and matches the predicted instances one-to-one to the requested
identity labels (IoU >= 0.5). Reports per-frame and mean "requested-instance F1" = 2TP / (2TP + FP + FN) where
TP = label instances matched by a prediction, FN = unmatched labels, FP = unmatched predictions.

Modes
  --serve                     persistent server: reads one JSON request per stdin line
                              {"frames_dir": ..., "labels_dir": ..., "scale": 1.0, "n_frames": 17}
                              and prints one JSON result per line (stdout). gen_corpus.py uses this so the
                              model is loaded once per corpus.
  --pairs FRAMES:LABELS ...   one-shot evaluation of explicit (frames_dir, labels_dir) pairs
  --clips_root R --sub S      one-shot: every R/clip_*/S (S = gen | target) against R/clip_*/labels
Options: --scale (upsample factor before segmentation, masks resized back with nearest), --gpu, --model (cpsam path),
         --flow_threshold, --cellprob_threshold, --min_size, --limit, --out JSON path.
"""
import os, sys, json, time, argparse, glob

ap = argparse.ArgumentParser()
ap.add_argument("--serve", action="store_true")
ap.add_argument("--pairs", nargs="*", default=[])
ap.add_argument("--clips_root", default=None)
ap.add_argument("--sub", default="gen")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--n_frames", type=int, default=0, help="0 = all frames found")
ap.add_argument("--gpu", default=None)
ap.add_argument("--model", default=os.environ.get("CELLPOSE_MODEL", os.path.expanduser("~/.cellpose/models/cpsam")))
ap.add_argument("--flow_threshold", type=float, default=0.4)
ap.add_argument("--cellprob_threshold", type=float, default=0.0)
ap.add_argument("--min_size", type=int, default=15)
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import cv2

_MODEL = None


def log(*a):
    print("[readout_f1]", *a, file=sys.stderr, flush=True)


def get_model():
    global _MODEL
    if _MODEL is None:
        t0 = time.time()
        from cellpose import models
        _MODEL = models.CellposeModel(gpu=True, pretrained_model=args.model)
        log(f"Cellpose-SAM loaded from {args.model} in {time.time() - t0:.1f}s")
    return _MODEL


def read_gray(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32).mean(-1)  # BGR order irrelevant for mean
    return a.astype(np.float32)


def read_label(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.int64)


def match_f1(gt, pred, iou_thr):
    """One-to-one matching at IoU >= iou_thr (unique by construction for thr >= 0.5). Returns dict."""
    gt = gt.astype(np.int64); pred = pred.astype(np.int64)
    ng, npd = int(gt.max()), int(pred.max())
    if ng == 0 and npd == 0:
        return dict(tp=0, fp=0, fn=0, f1=1.0, n_gt=0, n_pred=0)
    if ng == 0 or npd == 0:
        return dict(tp=0, fp=npd, fn=ng, f1=0.0, n_gt=ng, n_pred=npd)
    # overlap counts for pairs (g, p) with both nonzero
    both = (gt > 0) & (pred > 0)
    inter = np.bincount(gt[both] * (npd + 1) + pred[both], minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    area_g = np.bincount(gt.ravel(), minlength=ng + 1)[:, None]
    area_p = np.bincount(pred.ravel(), minlength=npd + 1)[None, :]
    union = area_g + area_p - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)[1:, 1:]
    if iou_thr >= 0.5:
        tp = int((iou >= iou_thr).sum())  # each row/col can hold at most one such entry
    else:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-iou)
        tp = int((iou[r, c] >= iou_thr).sum())
    n_gt = int((area_g[1:, 0] > 0).sum()); n_pred = int((area_p[0, 1:] > 0).sum())
    fp, fn = n_pred - tp, n_gt - tp
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, f1=float(f1), n_gt=n_gt, n_pred=n_pred)


def segment(frames, scale):
    model = get_model()
    imgs = []
    for f in frames:
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        imgs.append(f)
    masks, _, _ = model.eval(imgs, batch_size=args.batch_size, normalize=True, diameter=None,
                             flow_threshold=args.flow_threshold, cellprob_threshold=args.cellprob_threshold,
                             min_size=args.min_size)
    if not isinstance(masks, list):
        masks = [masks]
    out = []
    for m, f in zip(masks, frames):
        m = np.asarray(m)
        if m.shape != f.shape:
            m = cv2.resize(m.astype(np.int32), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        out.append(m.astype(np.int32))
    return out


def evaluate(frames_dir, labels_dir, scale=1.0, n_frames=0, iou_thr=None):
    iou_thr = args.iou if iou_thr is None else iou_thr
    t0 = time.time()
    ff = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    lf = sorted(glob.glob(os.path.join(labels_dir, "frame_*.png")))
    if n_frames:
        ff, lf = ff[:n_frames], lf[:n_frames]
    if not ff or len(ff) != len(lf):
        raise RuntimeError(f"frame/label count mismatch: {len(ff)} frames in {frames_dir}, {len(lf)} labels in {labels_dir}")
    frames = [read_gray(p) for p in ff]
    labels = [read_label(p) for p in lf]
    preds = segment(frames, scale)
    per = [match_f1(g, p, iou_thr) for g, p in zip(labels, preds)]
    f1s = [x["f1"] for x in per]
    return dict(frames_dir=frames_dir, labels_dir=labels_dir, n_frames=len(ff), scale=scale, iou=iou_thr,
                f1_mean=float(np.mean(f1s)), f1_min=float(np.min(f1s)), f1_per_frame=[round(v, 4) for v in f1s],
                tp=int(sum(x["tp"] for x in per)), fp=int(sum(x["fp"] for x in per)), fn=int(sum(x["fn"] for x in per)),
                n_gt_mean=float(np.mean([x["n_gt"] for x in per])), n_pred_mean=float(np.mean([x["n_pred"] for x in per])),
                seconds=round(time.time() - t0, 2))


def main():
    if args.serve:
        proto = sys.stdout            # protocol channel; all other prints (cellpose, tqdm) go to stderr
        sys.stdout = sys.stderr
        get_model()
        print("@@" + json.dumps(dict(ready=True)), file=proto, flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("quit"):
                    break
                res = evaluate(req["frames_dir"], req["labels_dir"], float(req.get("scale", args.scale)),
                               int(req.get("n_frames", args.n_frames)), req.get("iou"))
            except Exception as e:  # keep serving
                res = dict(error=f"{type(e).__name__}: {e}", frames_dir=req.get("frames_dir") if isinstance(req, dict) else None)
            print("@@" + json.dumps(res), file=proto, flush=True)
        return
    pairs = [tuple(p.split(":")) for p in args.pairs]
    if args.clips_root:
        for c in sorted(glob.glob(os.path.join(args.clips_root, "clip_*"))):
            if os.path.isdir(os.path.join(c, args.sub)):
                pairs.append((os.path.join(c, args.sub), os.path.join(c, "labels")))
    if args.limit:
        pairs = pairs[:args.limit]
    results = []
    for fd, ld in pairs:
        try:
            r = evaluate(fd, ld, args.scale, args.n_frames)
        except Exception as e:
            r = dict(frames_dir=fd, labels_dir=ld, error=f"{type(e).__name__}: {e}")
        log(json.dumps({k: r[k] for k in r if k not in ("f1_per_frame",)}))
        results.append(r)
    ok = [r["f1_mean"] for r in results if "f1_mean" in r]
    summary = dict(n=len(results), n_ok=len(ok), f1_mean_over_clips=float(np.mean(ok)) if ok else None,
                   f1_min_over_clips=float(np.min(ok)) if ok else None, scale=args.scale, results=results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in summary if k != "results"}))


if __name__ == "__main__":
    main()
