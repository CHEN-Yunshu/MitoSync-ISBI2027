#!/usr/bin/env python
"""MitoSync M3 certifier: frozen Cellpose-SAM readout -> per-division keep / re-time / drop -> resample clips whose divisions do
not survive -> certified CTC set + per-position readability r(phi) for the phase-lock feedback.

Run with the cellres env (cellpose 4, numpy):  PYC=CELLPOSE_PYTHON
  $PYC certify.py --dataset PhC-C2DL-PSC --clips_root <M1 clips (cond_v2|cond, ref.png, labels, meta.json)> \
      --twin <M1 CTC label twin (clip_XXX_GT/TRA, clip_XXX_ST/SEG)> --lora <VACE LoRA .safetensors> \
      --out_root $N/data/mitosync/cert/<tag> --gpu G [--clips clip_000,clip_001 | --limit K] [--tau 0.8] [--max_retries 2] \
      [--seed_base 0] [--seed_stride 1000] [--steps 30 --cfg 5] [--f1_thr .75] [--gen_existing <ctc root with t*.tif>] \
      [--on_exhausted best|drop] [--drop_gen_png] [--drop_failed_attempts] [--workers 4]

Per round r = 0..max_retries (round 0 = every clip; round r>0 = the clips that failed every earlier round):
  (--reassemble skips 1-2 and re-decides 3 from the recorded attempts, e.g. with another --f1_thr / --tau.)
  1. generation: scripts/vace/gen_corpus.py UNCHANGED (vace env subprocess, pipeline + LoRA loaded once per round) into
     <out_root>/_attempts/try<r>/{clips,ctc}; the attempt clip root holds a real directory per clip of the FULL clip list
     (inputs symlinked: cond -> cond_v2, ref.png, labels, meta.json, lineage.txt) so gen_corpus's seed rule
     seed = seed_base_r + index(clip in the sorted clip list) gives seed(clip, r) = seed_base + r*seed_stride + index(clip)
     ("a new seed" per retry, reproducible).  Round 0 may instead link existing generations (--gen_existing).
  2. readout: Cellpose-SAM (cpsam, CP_SETTINGS of relabel_by_readout.py) on the generated t%03d.tif ->
     try<r>/rl/<clip>/cpsam.npz; then relabel_by_readout.relabel_clip UNCHANGED: per programmed division (parent with two
     daughters born at t_b) t* = first t >= t_b with both daughters matched (IoU >= .5) at t and t+1:
     t* = t_b -> kept; t* > t_b -> re-timed (mother extended to t*-1, daughters born at t*); none -> dropped (daughters merged
     into the mother).  Writes the relabelled twin try<r>/rl/<clip>{,_GT,_ST} and runs ctc_io.verify_ctc.
  3. decision: survival s = (kept + re-timed) / programmed (programmed = divisions minus skipped_irregular; s = 1 if none);
     clip gate = verify_ok and requested-instance readout F1 (vs the programmed labels) >= --f1_thr (default .75 for
     PhC-C2DL-PSC, none otherwise, as relabel_by_readout.py); PASS iff gate and s >= --tau (0.8).
  A clip that never passes after max_retries resamples: --on_exhausted best (default) keeps the attempt with the highest s
  among gated attempts (its labels are still readout-consistent; flagged 'exhausted'), drop removes it.  A clip with no gated
  attempt is dropped.
Outputs (every attempt recorded):
  <out_root>/<clip>/t%03d.tif (symlinks to the chosen attempt's images), <clip>_GT/TRA + <clip>_ST/SEG (copied relabelled
  labels), <clip>_CP/SEG (Cellpose-SAM masks of the same frames, for the CP-SAM deployment variant);
  accepted_dirs.txt (certified clip dirs), certify_manifest.json (per clip: every attempt with seed, F1, kept / re-timed /
  dropped, s, per-division records with phi), certify_summary.json, readability_phi.json:
  r(phi) = kept-at-birth / programmed by latent position phi = (t_b - 1) mod 4 of the birth frame in the Wan VAE causal grid
  ({0}, {1-4}, {5-8}, {9-12}, {13-16}), over ALL attempts (each generation is an unselected sample: the estimate for the
  phase-lock feedback), over first attempts only, and over the final certified set (selected, for reference); Wilson 95 % CI,
  delay histograms, and a suggested --phase_lock set (positions whose r is >= 50 % of the best position; ranked by survival
  instead when kept-at-birth is too rare, max r < 0.15).
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N = Path(os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
sys.path.insert(0, str(N / 'scripts/vace'))   # relabel_by_readout / ctc_io (module level: spawn workers unpickle RL._worker)
PYV = os.environ.get('VACE_PYTHON', sys.executable)
GEN = N / 'scripts/vace/gen_corpus.py'
VERSION = 'certify v1 (2026-10-10)'
log = lambda *a: print(time.strftime('%Y-%m-%d %H:%M:%S'), '[certify]', *a, flush=True)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True); ap.add_argument('--clips_root', required=True); ap.add_argument('--twin', required=True)
    ap.add_argument('--lora', required=True); ap.add_argument('--out_root', required=True); ap.add_argument('--gpu', required=True)
    ap.add_argument('--clips', default=None); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.8); ap.add_argument('--max_retries', type=int, default=2)
    ap.add_argument('--seed_base', type=int, default=0); ap.add_argument('--seed_stride', type=int, default=1000)
    ap.add_argument('--steps', type=int, default=30); ap.add_argument('--cfg', type=float, default=5.0)
    ap.add_argument('--f1_thr', type=float, default=None); ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--gen_existing', default=None, help='round 0 uses <dir>/<clip>/t%%03d.tif instead of generating')
    ap.add_argument('--on_exhausted', choices=['best', 'drop'], default='best')
    ap.add_argument('--drop_gen_png', action='store_true', help='delete the generated PNG frames after the readout (disk)')
    ap.add_argument('--drop_failed_attempts', action='store_true', help='delete images of attempts that were not chosen (records kept)')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--min_free_gb', type=float, default=30); ap.add_argument('--gen_gb', type=float, default=22)
    ap.add_argument('--max_used_gb', type=float, default=88); ap.add_argument('--wait_gpu_s', type=int, default=6 * 3600)
    ap.add_argument('--gen_args', default='', help='extra args for gen_corpus.py')
    ap.add_argument('--reassemble', action='store_true', help='no generation / readout: recompute gate and PASS of the recorded attempts '
                    'with the current --tau / --f1_thr and rebuild the certified set (previous summary kept as certify_summary_prev_<time>.json)')
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------ gpu
def gpu_mem(g):
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', '-i', str(g)],
                         capture_output=True, text=True).stdout.strip().split(',')
    used, tot = float(out[0]) / 1024, float(out[1]) / 1024
    return used, tot


def wait_gpu(a, need_gb):
    t0 = time.time()
    while True:
        used, tot = gpu_mem(a.gpu)
        if tot - used >= a.min_free_gb and used + need_gb <= a.max_used_gb:
            return used
        if time.time() - t0 > a.wait_gpu_s:
            raise RuntimeError(f'GPU {a.gpu} never had {need_gb} GB headroom (used {used:.1f}/{tot:.1f})')
        if int(time.time() - t0) % 1800 < 60: log(f'waiting for GPU {a.gpu}: used {used:.1f}/{tot:.1f} GB, need {need_gb} GB below {a.max_used_gb}')
        time.sleep(60)


# ------------------------------------------------------------------------------------------------ attempt layout
def link(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def prepare_attempt(a, r, all_clips, todo):
    A = Path(a.out_root) / '_attempts' / f'try{r}'
    cdir, gdir, rdir = A / 'clips', A / 'ctc', A / 'rl'
    for d in (cdir, gdir, rdir): d.mkdir(parents=True, exist_ok=True)
    for c in all_clips: (cdir / c).mkdir(exist_ok=True)   # placeholders keep gen_corpus's index = index in the full list
    for c in todo:
        src = Path(a.clips_root) / c
        link(src / ('cond' if (src / 'cond').is_dir() else 'cond_v2'), cdir / c / 'cond')
        for f in ('ref.png', 'labels', 'meta.json', 'lineage.txt', 'events.json'):
            if (src / f).exists(): link(src / f, cdir / c / f)
        for sub in ('_GT', '_ST'):
            link(Path(a.twin) / f'{c}{sub}', gdir / f'{c}{sub}')
    return A, cdir, gdir, rdir


def generate(a, r, cdir, gdir, todo, all_clips):
    T = n_frames(a, todo[0])
    if r == 0 and a.gen_existing:
        for c in todo:
            (gdir / c).mkdir(exist_ok=True)
            for f in sorted(glob.glob(os.path.join(a.gen_existing, c, 't*.tif'))):
                link(f, gdir / c / os.path.basename(f))
        return dict(source='gen_existing', dir=os.path.realpath(a.gen_existing))
    seed_base = a.seed_base + r * a.seed_stride
    wait_gpu(a, a.gen_gb)
    lf = gdir.parent / 'gen.log'
    cmd = [PYV, str(GEN), '--dataset', a.dataset, '--lora', a.lora, '--clips_root', str(cdir), '--ctc_root', str(gdir), '--gpu', str(a.gpu),
           '--steps', str(a.steps), '--cfg', str(a.cfg), '--seed_base', str(seed_base), '--clips', ','.join(todo), '--num_frames', str(T), '--force']
    if a.gen_args: cmd += a.gen_args.split()
    log(f'round {r}: generating {len(todo)} clips, seed_base {seed_base}, log {lf}')
    t0 = time.time()
    with open(lf, 'a') as f:
        f.write(f'\n===== {time.ctime()} {" ".join(cmd)}\n'); f.flush()
        rc = subprocess.run(cmd, cwd=str(N), stdout=f, stderr=subprocess.STDOUT).returncode
    log(f'round {r}: gen_corpus rc {rc} in {time.time() - t0:.0f}s')
    return dict(source='gen_corpus', rc=rc, seed_base=seed_base, seconds=round(time.time() - t0, 1), log=str(lf),
                seeds={c: seed_base + all_clips.index(c) for c in todo})


def n_frames(a, clip):
    return len(glob.glob(os.path.join(a.twin, f'{clip}_ST', 'SEG', 'man_seg*.tif')))


# ------------------------------------------------------------------------------------------------ readout + relabel
_MODEL = None


def readout(a, RL, gdir, rdir, todo):
    global _MODEL
    if _MODEL is None:
        from cellpose import models
        _MODEL = models.CellposeModel(gpu=True, pretrained_model='cpsam')
    for c in todo:
        npz = rdir / c / 'cpsam.npz'
        if npz.exists() or not (gdir / c / 't000.tif').exists(): continue
        t0 = time.time(); shp, n = RL.segment_clip(_MODEL, str(gdir / c), str(npz), a.batch)
        log(f'  readout {c} {shp} cells/frame {np.mean(n):.1f} {time.time() - t0:.1f}s')


def free_model():
    global _MODEL
    _MODEL = None
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass


def phi_of(tb):
    return int((tb - 1) % 4) if tb >= 1 else None


def evaluate(a, RL, gdir, rdir, todo, rad, f1_thr, r, gen_info):
    import multiprocessing as mp
    jobs = [c for c in todo if (rdir / c / 'cpsam.npz').exists()]
    with mp.get_context('spawn').Pool(max(1, min(a.workers, len(jobs)))) as pool:
        res = dict(zip(jobs, pool.map(RL._worker, [(str(gdir / c), str(rdir), rad, a.iou, False) for c in jobs])))
    out = {}
    for c in todo:
        att = dict(attempt=r, seed=(gen_info.get('seeds') or {}).get(c), gen=gen_info.get('source'), images=str(gdir / c), rl=str(rdir / c))
        if c not in res:
            att.update(status='gen_or_readout_missing', passed=False); out[c] = att; continue
        x = res[c]
        if 'error' in x:
            att.update(status='relabel_error', error=x['error'], passed=False); out[c] = att; continue
        prog = x['divisions_original'] - x['skipped_irregular']; surv = x['kept_at_birth'] + x['delayed']
        s = surv / prog if prog else 1.0
        gate = bool(x['verify_ok']) and (f1_thr is None or x['readout_f1_orig'] >= f1_thr)
        divs = []
        for d in x['divisions']:
            d = dict(d); d['phi'] = phi_of(d['tb'])
            if d.get('t_star') is not None: d['phi_t_star'] = phi_of(d['t_star'])
            divs.append(d)
        att.update(status='ok', programmed=prog, kept=x['kept_at_birth'], retimed=x['delayed'], dropped=x['dropped'] + x['dropped_cascade'],
                   skipped_irregular=x['skipped_irregular'], survival=s, readout_f1_requested=x['readout_f1_orig'], readout_f1_relabelled=x['readout_f1_new'],
                   verify_ok=bool(x['verify_ok']), gate=gate, passed=bool(gate and s >= a.tau), delay_mean=x['delay_mean'], delays=x['delays'],
                   pixels_relabelled=x['pixels_relabelled'], bridge_failed_frames=x['bridge_failed_frames'], divisions=divs)
        out[c] = att
        log(f'  {c} try{r}: programmed {prog} kept {att["kept"]} retimed {att["retimed"]} dropped {att["dropped"]} s={s:.2f} '
            f'F1req={x["readout_f1_orig"]:.3f} verify={x["verify_ok"]} -> {"PASS" if att["passed"] else "FAIL"}')
    return out


# ------------------------------------------------------------------------------------------------ outputs
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def readability(attempts_by_clip, chosen, out_root):
    def agg(sel):
        cnt = {p: Counter() for p in range(4)}; dl = {p: Counter() for p in range(4)}
        for att in sel:
            for d in att.get('divisions', []):
                if d.get('phi') is None or d['outcome'] == 'skipped_irregular': continue
                o = {'kept': 'kept', 'delayed': 'retimed'}.get(d['outcome'], 'dropped')
                cnt[d['phi']][o] += 1
                if o == 'retimed': dl[d['phi']][int(d['delay'])] += 1
        res = {}
        for p in range(4):
            n = sum(cnt[p].values()); k = cnt[p]['kept']
            res[p] = dict(n=n, kept=k, retimed=cnt[p]['retimed'], dropped=cnt[p]['dropped'], r=(k / n if n else None), r_ci95=wilson(k, n),
                          survival=((k + cnt[p]['retimed']) / n if n else None), delay_hist=dict(sorted(dl[p].items())))
        return res
    every = [att for atts in attempts_by_clip.values() for att in atts if att.get('status') == 'ok']
    first = [atts[0] for atts in attempts_by_clip.values() if atts and atts[0].get('status') == 'ok']
    fin = [attempts_by_clip[c][i] for c, i in chosen.items() if i is not None]
    all_ = agg(every); rs = {p: v['r'] for p, v in all_.items() if v['r'] is not None}
    basis = 'r (kept-at-birth)'
    if not rs or max(rs.values()) < 0.15:   # kept-at-birth too rare to rank positions: rank by survival (kept + re-timed)
        rs = {p: v['survival'] for p, v in all_.items() if v['survival'] is not None}; basis = 'survival (kept + re-timed), since max r < 0.15'
    best = max(rs.values()) if rs else None
    out = dict(definition='phi = (t_b - 1) mod 4 of the programmed birth frame t_b (Wan VAE causal grid {0},{1-4},{5-8},{9-12},{13-16}); '
                          'r(phi) = kept-at-birth / programmed divisions (relabel_by_readout rule)',
               all_attempts=all_, first_attempts=agg(first), final_certified=agg(fin),
               r_hat=[all_[p]['r'] for p in range(4)],
               suggested_phase_lock=','.join(str(p) for p in sorted(rs) if best and rs[p] >= 0.5 * best), suggestion_basis=basis,
               survival_hat=[all_[p]['survival'] for p in range(4)], n_by_phi=[all_[p]['n'] for p in range(4)],
               note='all_attempts is the unselected estimate for the phase-lock feedback (lineage_sim_v3 --phase_lock / P(t_b) ~ r(phi)^beta); '
                    'final_certified is selected by the resampling and only for reference')
    json.dump(out, open(Path(out_root) / 'readability_phi.json', 'w'), indent=1)
    return out


def assemble(a, all_todo, attempts, rad):
    out = Path(a.out_root); chosen = {}
    # verifier fix 2026-10-10: an attempt whose images were deleted (--drop_failed_attempts in an earlier run) can never be
    # chosen again (e.g. by --reassemble with a lower --f1_thr): it would give an image-less clip dir in accepted_dirs.txt that
    # verify_ctc (labels only) does not catch.
    def has_images(x):
        return not x.get('images_deleted') and bool(glob.glob(os.path.join(x.get('images', ''), 't*.tif')))
    for c in all_todo:
        atts = attempts.get(c, [])
        skipped = [i for i, x in enumerate(atts) if (x.get('passed') or x.get('gate')) and not has_images(x)]
        if skipped: log(f'  {c}: attempts {skipped} pass the gate but their images were deleted -> not eligible')
        idx = next((i for i, x in enumerate(atts) if x.get('passed') and has_images(x)), None); status = 'certified'
        if idx is None:
            gated = [(x.get('survival', -1), x.get('readout_f1_requested', 0), i) for i, x in enumerate(atts) if x.get('gate') and has_images(x)]
            if gated and a.on_exhausted == 'best':
                idx = max(gated)[2]; status = 'exhausted_best'
            else:
                status = 'dropped_no_gated_attempt' if not gated else 'dropped_exhausted'
        chosen[c] = idx
        for sub in ('', '_GT', '_ST', '_CP'):
            p = out / f'{c}{sub}'
            if os.path.islink(p): os.remove(p)
            elif p.exists(): shutil.rmtree(p)
        if idx is None:
            for x in atts: x['chosen'] = False
            attempts[c + '__status'] = status; continue
        att = atts[idx]; rl = Path(att['rl']); gd = Path(att['images'])
        (out / c).mkdir()
        for f in sorted(gd.glob('t*.tif')): link(f, out / c / f.name)
        shutil.copytree(rl.parent / f'{c}_GT', out / f'{c}_GT'); shutil.copytree(rl.parent / f'{c}_ST', out / f'{c}_ST')
        import tifffile
        cp = np.load(rl / 'cpsam.npz')['masks']; (out / f'{c}_CP' / 'SEG').mkdir(parents=True)
        for t in range(cp.shape[0]): tifffile.imwrite(out / f'{c}_CP' / 'SEG' / f'man_seg{t:03d}.tif', cp[t].astype(np.uint16), compression='zlib')
        for i, x in enumerate(atts): x['chosen'] = (i == idx)
        attempts[c + '__status'] = status
    return chosen


def main():
    a = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)   # cellpose in this process; gen_corpus sets its own
    import relabel_by_readout as RL
    from ctc_io import verify_ctc
    out = Path(a.out_root).resolve(); a.out_root = str(out)
    assert str(out).startswith(str(N / 'data/mitosync') + '/'), 'out_root must be a new path under $N/data/mitosync/'
    out.mkdir(parents=True, exist_ok=True)
    rad = RL.MARKER_R[a.dataset]
    f1_thr = a.f1_thr if a.f1_thr is not None else (0.75 if a.dataset == 'PhC-C2DL-PSC' else None)
    all_clips = sorted(d for d in os.listdir(a.clips_root) if d.startswith('clip_') and os.path.isdir(os.path.join(a.clips_root, d)))
    todo = [c for c in all_clips if os.path.exists(os.path.join(a.twin, f'{c}_GT', 'TRA', 'man_track.txt'))]
    if a.clips: todo = [c for c in todo if c in set(a.clips.split(','))]
    if a.limit: todo = todo[:a.limit]
    log(f'{a.dataset}: {len(todo)} clips ({len(all_clips)} in clip root); tau {a.tau}, max_retries {a.max_retries}, f1_thr {f1_thr}, '
        f'lora {a.lora}, gpu {a.gpu}, out {out}')
    mf = out / 'certify_manifest.json'
    man = json.load(open(mf)) if mf.exists() else {}
    attempts = {c: man.get('clips', {}).get(c, {}).get('attempts', []) for c in todo}
    rounds = man.get('rounds', [])
    if a.reassemble:
        for c in todo:
            for x in attempts[c]:
                if x.get('status') == 'ok':
                    x['gate'] = bool(x['verify_ok']) and (f1_thr is None or x['readout_f1_requested'] >= f1_thr)
                    x['passed'] = bool(x['gate'] and x['survival'] >= a.tau)
        if (out / 'certify_summary.json').exists():
            shutil.copy(out / 'certify_summary.json', out / f'certify_summary_prev_{time.strftime("%Y%m%d_%H%M%S")}.json')
        log(f'reassemble with tau {a.tau}, f1_thr {f1_thr}: {sum(1 for c in todo if any(x.get("passed") for x in attempts[c]))}/{len(todo)} clips have a passing attempt')
    for r in range(0 if not a.reassemble else a.max_retries + 1, a.max_retries + 1):
        pend = [c for c in todo if not any(x.get('passed') for x in attempts[c]) and not any(x.get('attempt') == r for x in attempts[c])
                and len(attempts[c]) == r]
        if not pend: continue
        A, cdir, gdir, rdir = prepare_attempt(a, r, all_clips, pend)
        free_model()
        gi = generate(a, r, cdir, gdir, pend, all_clips)
        wait_gpu(a, 6)
        readout(a, RL, gdir, rdir, pend)
        res = evaluate(a, RL, gdir, rdir, pend, rad, f1_thr, r, gi)
        for c in pend: attempts[c].append(res[c])
        if a.drop_gen_png:
            for c in pend: shutil.rmtree(cdir / c / 'gen', ignore_errors=True)
        rounds.append(dict(round=r, clips=pend, gen=gi, passed=[c for c in pend if res[c].get('passed')], time=time.strftime('%Y-%m-%d %H:%M:%S')))
        man.update(clips={c: dict(attempts=attempts[c]) for c in todo}, rounds=rounds)
        json.dump(man, open(mf, 'w'), indent=1)
        log(f'round {r}: {len(rounds[-1]["passed"])}/{len(pend)} passed')
    free_model()
    chosen = assemble(a, todo, attempts, rad)
    status = {c: attempts.pop(c + '__status') for c in todo}
    if a.drop_failed_attempts:
        for c in todo:
            for x in attempts[c]:
                if not x.get('chosen'):
                    for f in glob.glob(os.path.join(x['images'], 't*.tif')): os.remove(f)
                    x['images_deleted'] = True
    acc = [str(out / c) for c in todo if chosen[c] is not None]
    (out / 'accepted_dirs.txt').write_text(''.join(d + '\n' for d in acc))
    ver = {c: verify_ctc(str(out / c)) for c in todo if chosen[c] is not None}
    rph = readability(attempts, chosen, out)
    fin = [attempts[c][chosen[c]] for c in todo if chosen[c] is not None]
    every = [x for c in todo for x in attempts[c] if x.get('status') == 'ok']
    def tot(sel, k): return int(sum(x.get(k, 0) for x in sel))
    summ = dict(dataset=a.dataset, version=VERSION, reassembled=bool(a.reassemble), script=os.path.realpath(__file__), lora=a.lora, steps=a.steps, cfg=a.cfg, tau=a.tau,
                max_retries=a.max_retries, f1_thr=f1_thr, seed_base=a.seed_base, seed_stride=a.seed_stride, clips_root=os.path.realpath(a.clips_root),
                twin=os.path.realpath(a.twin), n_clips=len(todo), status=dict(Counter(status.values())), per_clip_status=status,
                attempts_total=len(every), attempts_per_clip=dict(Counter(len(attempts[c]) for c in todo)),
                certified=len(acc), verify_ok_final=int(sum(1 for v in ver.values() if v['bad_frames'] == 0 and v['bad_parents'] == 0 and v['multi_component'] == 0)),
                final=dict(programmed=tot(fin, 'programmed'), kept=tot(fin, 'kept'), retimed=tot(fin, 'retimed'), dropped=tot(fin, 'dropped'),
                           divisions_after_certification=int(sum(v['divisions'] for v in ver.values()))),
                all_attempts=dict(programmed=tot(every, 'programmed'), kept=tot(every, 'kept'), retimed=tot(every, 'retimed'), dropped=tot(every, 'dropped')),
                first_attempt_pass_rate=(sum(1 for c in todo if attempts[c] and attempts[c][0].get('passed')) / max(len(todo), 1)),
                r_hat_phi=rph['r_hat'], suggested_phase_lock=rph['suggested_phase_lock'], updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    man.update(clips={c: dict(attempts=attempts[c], chosen=chosen[c], status=status[c]) for c in todo}, rounds=rounds, summary=summ)
    json.dump(man, open(mf, 'w'), indent=1)
    json.dump(summ, open(out / 'certify_summary.json', 'w'), indent=1)
    log('SUMMARY', json.dumps({k: summ[k] for k in ('n_clips', 'status', 'attempts_total', 'certified', 'verify_ok_final', 'final', 'all_attempts', 'r_hat_phi', 'suggested_phase_lock')}))


if __name__ == '__main__':
    main()
