#!/usr/bin/env python3
"""Amendment 10 Part B: build 3-frame 48x48 division-detection candidates per source.

Writes only beneath $N/data/mitosync/pixel_div_20261018 and $N/eval/mitosync/pixel_div_20261018.
Details fixed in DESIGN_DETAILS.md (sections 1-3) before this script ran.
"""
import argparse, collections, json, os, sys, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tifffile

N = Path('STUDY_ROOT')
CACHE = N / 'data/mitosync/pixel_div_20261018'
OUT = N / 'eval/mitosync/pixel_div_20261018'
HALF, CROP = 24, 48
P_LO, P_HI = 1.0, 99.5


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def log(*a):
    print(now(), *a, flush=True)


def norm_u8(x):
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, [P_LO, P_HI])
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.floor(255.0 * y + 0.5).astype(np.uint8), float(lo), float(hi)


def frame_files(d, prefix):
    fs = sorted(Path(d).glob(prefix + '*.tif'))
    idx = [int(f.name[len(prefix):-4]) for f in fs]
    assert idx == list(range(len(fs))), (d, prefix, idx[:5], len(fs))
    return fs


def read_tracks(p):
    a = np.loadtxt(p, dtype=np.int64, ndmin=2)
    tr = {int(r[0]): (int(r[1]), int(r[2]), int(r[3])) for r in a}
    ch = collections.defaultdict(list)
    for L, (B, E, P) in tr.items():
        if P:
            ch[P].append(L)
    return tr, ch


def events_of(tr, ch, T):
    ev = []
    stats = collections.Counter()
    for P in sorted(ch):
        cs = sorted(ch[P])
        stats[f'children_{len(cs)}'] += 1
        if P not in tr:
            stats['missing_parent'] += 1
            continue
        if len(cs) != 2:
            stats['nonbinary_parent'] += 1
            continue
        tstar = tr[P][1]
        if not all(tr[c][0] == tstar + 1 for c in cs):
            stats['binary_bad_timing'] += 1
            continue
        stats['binary_timed'] += 1
        if 1 <= tstar <= T - 2:
            ev.append((P, tstar, cs[0], cs[1]))
        else:
            stats['binary_timed_out_of_range'] += 1
    stats['events'] = len(ev)
    return ev, dict(stats)


def process_video(job):
    name, img_dir, seg_dir, tra_dir, out_path = job
    imgs = frame_files(img_dir, 't')
    segs = frame_files(seg_dir, 'man_seg')
    tras = frame_files(tra_dir, 'man_track')
    T = len(imgs)
    assert len(segs) == T and len(tras) == T, (name, T, len(segs), len(tras))
    tr, ch = read_tracks(Path(tra_dir) / 'man_track.txt')
    ev, evstats = events_of(tr, ch, T)
    ev_mother = {P: i for i, (P, ts, d1, d2) in enumerate(ev)}
    ev_daughter = {}
    for i, (P, ts, d1, d2) in enumerate(ev):
        ev_daughter[d1] = i
        ev_daughter[d2] = i
    frames, lohi = [], []
    for f in imgs:
        a = tifffile.imread(f)
        assert a.ndim == 2, (f, a.shape)
        u, lo, hi = norm_u8(a)
        frames.append(np.pad(u, HALF, mode='reflect'))
        lohi.append((lo, hi))
    H, W = frames[0].shape[0] - 2 * HALF, frames[0].shape[1] - 2 * HALF
    rec = collections.defaultdict(list)
    crops = []
    n_unmatched = 0
    n_conflict = 0
    for t in range(1, T - 1):
        seg = tifffile.imread(segs[t]).astype(np.int64)
        tra = tifffile.imread(tras[t]).astype(np.int64)
        assert seg.shape == (H, W) and tra.shape == (H, W), (name, t, seg.shape, tra.shape, (H, W))
        ids = np.unique(seg)
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        rr, cc = np.nonzero(seg)
        lab = seg[rr, cc]
        cnt = np.bincount(lab, minlength=ids.max() + 1).astype(np.float64)
        sr = np.bincount(lab, weights=rr, minlength=ids.max() + 1)
        sc = np.bincount(lab, weights=cc, minlength=ids.max() + 1)
        # overlap seg id x tra id
        m = (seg > 0) & (tra > 0)
        pairs = collections.defaultdict(int)
        if m.any():
            key = seg[m] * (1 << 32) + tra[m]
            u, c = np.unique(key, return_counts=True)
            best = {}
            for k, n in zip(u.tolist(), c.tolist()):
                s, q = k >> 32, k & 0xFFFFFFFF
                if s not in best or n > best[s][1] or (n == best[s][1] and q < best[s][0]):
                    best[s] = (q, n)
        else:
            best = {}
        # one-to-one: each tra id goes to the seg object with largest overlap (ties: lower seg id)
        owner = {}
        for s in sorted(best):
            q, n = best[s]
            if q not in owner or n > best[owner[q]][1]:
                owner[q] = s
        trk_of = {}
        for s in ids.tolist():
            if s in best and owner[best[s][0]] == s:
                trk_of[s] = best[s][0]
            else:
                trk_of[s] = 0
                if s in best:
                    n_conflict += 1
                else:
                    n_unmatched += 1
        f0, f1, f2 = frames[t - 1], frames[t], frames[t + 1]
        for s in ids.tolist():
            cy = int(np.floor(sr[s] / cnt[s] + 0.5))
            cx = int(np.floor(sc[s] / cnt[s] + 0.5))
            k = trk_of[s]
            y = 0
            es = -1
            if k in ev_mother and ev[ev_mother[k]][1] == t:
                y = 1
                es = ev_mother[k]
            tol = []
            if k in ev_mother and t in (ev[ev_mother[k]][1] - 1, ev[ev_mother[k]][1]):
                tol.append(ev_mother[k])
            if k in ev_daughter and t == ev[ev_daughter[k]][1] + 1:
                tol.append(ev_daughter[k])
            tol = sorted(set(tol)) + [-1, -1]
            crops.append(np.stack([f0[cy:cy + CROP, cx:cx + CROP], f1[cy:cy + CROP, cx:cx + CROP],
                                   f2[cy:cy + CROP, cx:cx + CROP]]))
            rec['t'].append(t); rec['cy'].append(cy); rec['cx'].append(cx); rec['seg_id'].append(s)
            rec['area'].append(int(cnt[s])); rec['trk'].append(k); rec['y'].append(y); rec['ev_strict'].append(es)
            rec['ev_tol1'].append(tol[0]); rec['ev_tol2'].append(tol[1])
    n = len(crops)
    crops = np.stack(crops).astype(np.uint8) if n else np.zeros((0, 3, CROP, CROP), np.uint8)
    arr = {k: np.asarray(v, dtype=np.int32) for k, v in rec.items()}
    # unit checks
    y = arr['y']
    es = arr['ev_strict']
    pos_ev = es[y == 1]
    assert len(np.unique(pos_ev)) == len(pos_ev), (name, 'event with two strict candidates')
    key = arr['trk'].astype(np.int64) * 100000 + arr['t']
    mk = arr['trk'] > 0
    assert len(np.unique(key[mk])) == mk.sum(), (name, 'duplicate (track, t) among candidates')
    tol_any = (arr['ev_tol1'] >= 0) | (arr['ev_tol2'] >= 0)
    ev_with_tol = len(set(arr['ev_tol1'][arr['ev_tol1'] >= 0].tolist()) | set(arr['ev_tol2'][arr['ev_tol2'] >= 0].tolist()))
    np.savez_compressed(out_path, crops=crops, events=np.asarray(ev, dtype=np.int32).reshape(-1, 4),
                        frame_lohi=np.asarray(lohi, dtype=np.float64), T=np.int32(T), **arr)
    summ = dict(name=name, img_dir=str(img_dir), seg_dir=str(seg_dir), tra_dir=str(tra_dir),
                img_dir_resolved=str(Path(img_dir).resolve()), seg_dir_resolved=str(Path(seg_dir).resolve()),
                tra_dir_resolved=str(Path(tra_dir).resolve()), T=T, H=H, W=W,
                candidates=n, positives=int(y.sum()), tol_candidates=int(tol_any.sum()),
                events=len(ev), events_with_strict_candidate=int(len(pos_ev)), events_with_tol_candidate=ev_with_tol,
                st_unmatched=n_unmatched, st_conflict=n_conflict, event_stats=evstats,
                shard=str(out_path), shard_bytes=Path(out_path).stat().st_size,
                input_dtype=str(tifffile.imread(imgs[0]).dtype))
    return summ


def real_job(src, root, vid):
    root = Path(root)
    return (f'{src}/{vid}', root / vid, root / f'{vid}_ST/SEG', root / f'{vid}_GT/TRA', CACHE / src / f'{vid}.npz')


def clip_jobs(src, list_file):
    dirs = [Path(l.strip()) for l in Path(list_file).read_text().splitlines() if l.strip()]
    jobs = []
    for d in dirs:
        jobs.append((f'{src}/{d.name}', d, Path(str(d) + '_ST') / 'SEG', Path(str(d) + '_GT') / 'TRA',
                     CACHE / src / f'{d.name}.npz'))
    return dirs, jobs


SOURCES = {
    'psc': dict(
        real=[('psc_head', N / 'data/ctc/PhC-C2DL-PSC_trainhead', '02'),
              ('psc_tail', N / 'data/ctc/PhC-C2DL-PSC_valtail', '02'),
              ('psc_01', N / 'data/ctc/PhC-C2DL-PSC', '01')],
        syn=[('psc_G', N / 'data/mitosync/design/PhC-C2DL-PSC/i1_k1/accepted_dirs.txt'),
             ('psc_Raw', N / 'data/mitosync/audit_20261012/RawMatched/accepted_dirs.txt'),
             ('psc_Paint', N / 'data/mitosync/audit_20261012/PaintMatched/accepted_dirs.txt')]),
    'fluo': dict(
        real=[('fluo_head', N / 'data/ctc/Fluo-N2DL-HeLa_trainhead', '02'),
              ('fluo_tail', N / 'data/ctc/Fluo-N2DL-HeLa_valtail', '02'),
              ('fluo_01', N / 'data/ctc/Fluo-N2DL-HeLa', '01')],
        syn=[('fluo_G', N / 'data/mitosync/design/Fluo-N2DL-HeLa/i1_k1/accepted_dirs.txt')]),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pairing_check_psc():
    g = [Path(l.strip()) for l in SOURCES['psc']['syn'][0][1].read_text().splitlines() if l.strip()]
    r = [Path(l.strip()) for l in SOURCES['psc']['syn'][1][1].read_text().splitlines() if l.strip()]
    p = [Path(l.strip()) for l in SOURCES['psc']['syn'][2][1].read_text().splitlines() if l.strip()]
    assert len(g) == len(r) == len(p) == 191, (len(g), len(r), len(p))
    assert [x.name for x in g] == [x.name for x in r] == [x.name for x in p], 'clip order differs'
    errs = []
    for a, b, c in zip(g, r, p):
        if (b / 't000.tif').resolve() != (a / 't000.tif').resolve():
            errs.append([a.name, 'raw images differ'])
        for suf in ['_GT', '_ST']:
            if Path(str(c) + suf).resolve() != Path(str(a) + suf).resolve():
                errs.append([a.name, 'paint labels differ', suf])
        if (c / 't000.tif').resolve() == (a / 't000.tif').resolve():
            errs.append([a.name, 'paint images identical to G'])
    assert not errs, errs[:5]
    return dict(n=len(g), same_order=True, raw_images_resolve_to_G=True, paint_labels_resolve_to_G=True,
                paint_images_differ=True, lists={k: dict(path=str(v), sha256=sha(v)) for k, v in
                                                 [('G', SOURCES['psc']['syn'][0][1]), ('Raw', SOURCES['psc']['syn'][1][1]),
                                                  ('Paint', SOURCES['psc']['syn'][2][1])]})


def split_check(ds):
    full = N / 'data/ctc' / ds / '02'
    head = N / 'data/ctc' / f'{ds}_trainhead' / '02'
    tail = N / 'data/ctc' / f'{ds}_valtail' / '02'
    nh = len(list(head.glob('t*.tif')))
    nt = len(list(tail.glob('t*.tif')))
    nf = len(list(full.glob('t*.tif')))
    assert nh + nt == nf, (ds, nh, nt, nf)
    out = dict(full=str(full), head=str(head), tail=str(tail), n_head=nh, n_tail=nt, n_full=nf, checked=[])
    for t in [0, nh // 2, nh - 1]:
        same = np.array_equal(tifffile.imread(head / f't{t:03d}.tif'), tifffile.imread(full / f't{t:03d}.tif'))
        assert same, (ds, 'head', t)
        out['checked'].append(['head', t, t])
    for t in [0, nt - 1]:
        same = np.array_equal(tifffile.imread(tail / f't{t:03d}.tif'), tifffile.imread(full / f't{t + nh:03d}.tif'))
        assert same, (ds, 'tail', t)
        out['checked'].append(['tail', t, t + nh])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['psc', 'fluo'], required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: clips per synthetic source')
    a = ap.parse_args()
    cfg = SOURCES[a.dataset]
    report = dict(updated=now(), dataset=a.dataset, normalisation=f'per-frame percentile ({P_LO}, {P_HI}) -> uint8',
                  crop=f'{CROP}x{CROP}, reflect pad {HALF}', sources={})
    report['split_check'] = split_check('PhC-C2DL-PSC' if a.dataset == 'psc' else 'Fluo-N2DL-HeLa')
    if a.dataset == 'psc':
        report['pairing_check'] = pairing_check_psc()
    jobs_by_src = {}
    for src, root, vid in cfg['real']:
        jobs_by_src[src] = [real_job(src, root, vid)]
    for src, lf in cfg['syn']:
        dirs, jobs = clip_jobs(src, lf)
        if a.limit:
            jobs = jobs[:a.limit]
        jobs_by_src[src] = jobs
    if a.only:
        jobs_by_src = {k: v for k, v in jobs_by_src.items() if k in a.only.split(',')}
    alljobs = []
    for src, jobs in jobs_by_src.items():
        (CACHE / src).mkdir(parents=True, exist_ok=True)
        alljobs += jobs
    log('jobs', len(alljobs), {k: len(v) for k, v in jobs_by_src.items()})
    # big real videos first
    alljobs.sort(key=lambda j: 0 if j[0].split('/')[0].endswith(('_01', '_head', '_tail')) else 1)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(process_video, alljobs, chunksize=1):
            res[s['name']] = s
            if s['name'].split('/')[0].endswith(('_01', '_head', '_tail')) or len(res) % 50 == 0:
                log('done', s['name'], s['candidates'], s['positives'], s['events'])
    for src, jobs in jobs_by_src.items():
        rows = [res[j[0]] for j in jobs]
        agg = dict(n_videos=len(rows), candidates=sum(r['candidates'] for r in rows),
                   positives=sum(r['positives'] for r in rows), events=sum(r['events'] for r in rows),
                   events_with_strict_candidate=sum(r['events_with_strict_candidate'] for r in rows),
                   events_with_tol_candidate=sum(r['events_with_tol_candidate'] for r in rows),
                   tol_candidates=sum(r['tol_candidates'] for r in rows),
                   st_unmatched=sum(r['st_unmatched'] for r in rows), st_conflict=sum(r['st_conflict'] for r in rows),
                   shard_bytes=sum(r['shard_bytes'] for r in rows), event_stats=dict(sum((collections.Counter(r['event_stats']) for r in rows), collections.Counter())),
                   input_dtypes=sorted(set(r['input_dtype'] for r in rows)))
        agg['strict_coverage'] = agg['events_with_strict_candidate'] / max(1, agg['events'])
        agg['unit_test_positives_eq_events_with_candidate'] = agg['positives'] == agg['events_with_strict_candidate']
        report['sources'][src] = dict(summary=agg, videos=rows)
        log(src, json.dumps({k: v for k, v in agg.items() if k != 'event_stats'}), agg['event_stats'])
    tag = a.dataset + ('_smoke' if a.limit else '') + ('_' + a.only.replace(',', '_') if a.only else '')
    p = OUT / f'candidates_{tag}.json'
    p.write_text(json.dumps(report, indent=1))
    log('wrote', p)


if __name__ == '__main__':
    main()
