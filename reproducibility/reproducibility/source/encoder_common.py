"""Shared helpers for Amendment 10 Part B (pixel division detection)."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

N = Path('STUDY_ROOT')
CACHE = N / 'data/mitosync/pixel_div_20261018'
OUT = N / 'eval/mitosync/pixel_div_20261018'

ARMS = {'psc': {'R': [], 'RG': ['psc_G'], 'RRaw': ['psc_Raw'], 'RPaint': ['psc_Paint']},
        'fluo': {'R': [], 'RG': ['fluo_G']}}
ARM_LABEL = {'R': 'R', 'RG': 'R+G', 'RRaw': 'R+Raw', 'RPaint': 'R+Paint'}
REAL = {'psc': 'psc_head', 'fluo': 'fluo_head'}
EVAL = {'psc': ('psc_01', 'psc_tail'), 'fluo': ('fluo_01', 'fluo_tail')}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def expected_shards(dataset, src):
    rep = json.loads((OUT / f'candidates_{dataset}.json').read_text())
    return [Path(v['shard']) for v in rep['sources'][src]['videos']], rep['sources'][src]['summary']


def load_source(dataset, src, keys=('crops', 'y')):
    shards, summ = expected_shards(dataset, src)
    out = {k: [] for k in keys}
    for f in shards:
        d = np.load(f)
        for k in keys:
            out[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in out.items()}
    assert len(out['y']) == summ['candidates'] and int(out['y'].sum()) == summ['positives'], (src, len(out['y']))
    return out, summ


def ap_strict(y, s):
    return float(average_precision_score(y, s))


def ap_tol(s, ev1, ev2):
    """+/-1 frame AP: each event credited once by its best-scoring candidate in its tolerance set (DESIGN 5)."""
    n = len(s)
    in_set = (ev1 >= 0) | (ev2 >= 0)
    keep = ~in_set
    lab = np.zeros(n, np.int8)
    members = {}
    for i in np.nonzero(in_set)[0]:
        for e in (ev1[i], ev2[i]):
            if e >= 0:
                members.setdefault(int(e), []).append(int(i))
    used = set()
    uncredited = 0
    for e in sorted(members):
        idx = [i for i in members[e] if i not in used]
        if not idx:
            uncredited += 1
            continue
        idx = np.asarray(idx)
        b = int(idx[np.argmax(s[idx])])
        used.add(b)
        lab[b] = 1
        keep[b] = True
    return float(average_precision_score(lab[keep], s[keep])), dict(events=len(members), uncredited=uncredited,
                                                                   kept=int(keep.sum()), excluded=int((~keep).sum()))


def tail_threshold(y, s):
    p, r, thr = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)[:-1]
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def f1_at(y, s, tau):
    pred = s >= tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return dict(f1=float(f1), precision=float(p), recall=float(r), tp=tp, fp=fp, fn=fn)
