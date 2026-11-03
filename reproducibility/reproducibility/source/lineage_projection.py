"""Observed-frame-only image evidence and full-reference temporal projection."""
from pathlib import Path
import collections
import numpy as np


def frames(total, gap, offset):
    assert gap in (1, 8) and 0 <= offset < gap and total > offset
    return list(range(offset, total, gap))


def project_tracks(rows, kept):
    kept = np.asarray(kept, dtype=np.int64)
    assert kept.ndim == 1 and len(kept) and np.all(np.diff(kept) > 0)
    projected = {}
    for label, begin, end, parent in np.asarray(rows, dtype=np.int64).reshape(-1, 4):
        first = int(np.searchsorted(kept, begin))
        last = int(np.searchsorted(kept, end, side='right')) - 1
        if first <= last:
            projected[int(label)] = [first, last, int(parent)]
    for values in projected.values():
        if values[2] not in projected:
            values[2] = 0
    return np.array([[k, *v] for k, v in sorted(projected.items())], np.int64).reshape(-1, 4)


def norm_u8(x):
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, [1., 99.5])
    return np.floor(255 * np.clip((x-lo)/max(hi-lo, 1e-6), 0, 1)+.5).astype(np.uint8)


def observable_features(image_paths, mask_paths, encoder, node_geometry, crop_masks, reader=None):
    """No reference lineage or unobserved image is accepted by this API."""
    import torch
    if reader is None:
        from tifffile import imread
        reader = imread
    assert len(image_paths) == len(mask_paths) >= 3
    images = [np.pad(norm_u8(reader(p)), 24, mode='reflect') for p in image_paths]
    records, embeddings = collections.defaultdict(list), []
    for t in range(1, len(image_paths)-1):
        seg = reader(mask_paths[t]).astype(np.int64)
        yy, xx = np.nonzero(seg)
        if not len(yy):
            continue
        labels = seg[yy, xx]
        count = np.bincount(labels)
        sy = np.bincount(labels, weights=yy)
        sx = np.bincount(labels, weights=xx)
        ids = np.flatnonzero(count)
        cy = np.floor(sy[ids]/count[ids]+.5).astype(np.int32)
        cx = np.floor(sx[ids]/count[ids]+.5).astype(np.int32)
        for start in range(0, len(ids), 128):
            stop = start+128
            pixels = np.stack([np.stack([images[k][y:y+48, x:x+48]
                              for k in [t-1, t, t+1]]) for y, x in zip(cy[start:stop], cx[start:stop])])
            roi = crop_masks(seg, ids[start:stop], cy[start:stop], cx[start:stop])
            with torch.inference_mode():
                embeddings.append(encoder(torch.from_numpy(pixels), torch.from_numpy(roi)).numpy().astype(np.float16))
        for key, values in [('t', np.full(len(ids), t)), ('cy', cy), ('cx', cx),
                            ('area', count[ids]), ('seg_id', ids)]:
            records[key].extend(map(int, values))
    result = {k: np.asarray(v, np.int32) for k, v in records.items()}
    assert len(embeddings), 'No interior observed instances; preserve the failure'
    result['embedding'] = np.concatenate(embeddings)
    geometry = node_geometry({k: result[k] for k in ['t', 'cy', 'cx', 'area']})
    result.update(node_geometry_full=geometry, node_geometry=geometry.astype(np.float16))
    return result


def aggregate(rows, metrics):
    """All eight offsets within each seed first; no view becomes an extra seed."""
    groups = {}
    for dataset in ['psc', 'fluo']:
        for arm in ['reference', 'Geo', 'R', 'RG', 'PM', 'Raw']:
            for gap in [1, 8]:
                rr = [r for r in rows if (r['dataset'], r['arm'], r['gap']) == (dataset, arm, gap)]
                expected = {(s, o) for s in range(1, 6) for o in range(gap)}
                assert len(rr) == len(expected) and {(r['seed'], r['offset']) for r in rr} == expected
                group = {}
                for metric in metrics:
                    values = [float(np.mean([r['metrics'][metric] for r in rr if r['seed'] == s]))
                              for s in range(1, 6)]
                    group[metric] = dict(values=values, mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
                groups[f'{dataset}/{arm}/g{gap}'] = group
    return groups
