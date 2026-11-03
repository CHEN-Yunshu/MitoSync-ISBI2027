"""All-model barrier, observed-view inference and fixed 540-result evaluation."""
import argparse
import collections
import fcntl
import os
import subprocess
import sys
import time
import numpy as np
import tifffile
import torch
from common import *
from views import frames, project_tracks, observable_features, aggregate
from encoder import Encoder, crop_masks
from targets import node_geometry, pair_geometry
from model import ParentModel
from inference import score_graph
from export_ctc import export
import capacity
import association_decoder


def prepare():
    initial = read(ROOT / 'EVALUATION_SOURCE_SEALED.json')
    for p, h in initial['sources'].items():
        assert sha(p) == h, p
    inputs = {}
    for ds, train in TRAIN.items():
        barrier = read(train / 'ALL_MODELS_TRAINED.json')
        assert barrier['models'] == 25 and barrier['updates_each'] == 6000
        assert barrier['training_sha256'] == sha(train / 'TRAINING_SEALED.json')
        for p, h in barrier['bindings'].items():
            assert sha(p) == h, p
            inputs[p] = h
        for name in ['ALL_MODELS_TRAINED.json', 'TRAINING_SEALED.json', 'PRETRAIN_VERIFIED.json', 'scaling.npz']:
            inputs[str(train / name)] = sha(train / name)
        inputs[str(encoder_path(ds))] = sha(encoder_path(ds))
    assert read(TRAIN['psc'] / 'VERIFIED.json')['status'] == 'VERIFIED'
    # This durable receipt precedes every new video-01 file read in this study.
    put(ROOT / 'ALL_50_MODELS_BOUND.json', dict(time=time.time(), models=50, inputs=inputs))
    views = {}
    for ds, full_name in DATASETS.items():
        folder = N / 'data/ctc' / full_name
        paths = sorted((folder / '01').glob('t*.tif'))
        assert [p.name for p in paths] == [f't{t:03d}.tif' for t in range(len(paths))]
        assert len(paths) == (300 if ds == 'psc' else 92)
        for seed in range(1, 6):
            for name in ['model.pt', 'config.yaml', 'train_config.yaml']:
                p = tracker_path(ds, seed) / name
                inputs[str(p)] = sha(p)
        track = folder / '01_GT/TRA/man_track.txt'
        inputs[str(track)] = sha(track)
        for gap, offset in [(1, 0)] + [(8, i) for i in range(8)]:
            kept = frames(len(paths), gap, offset)
            images = [str(paths[i]) for i in kept]
            masks = [str(folder / '01_ST/SEG' / f'man_seg{i:03d}.tif') for i in kept]
            gt_masks = [str(folder / '01_GT/TRA' / f'man_track{i:03d}.tif') for i in kept]
            for p in images + masks + gt_masks:
                if p not in inputs:
                    inputs[p] = sha(p)
            views[view_key(ds, gap, offset)] = dict(dataset=ds, gap=gap, offset=offset,
                frames=len(kept), source_frames=kept, images=images, masks=masks,
                gt_masks=gt_masks, track_file=str(track), neutral_times=[0, len(kept)-1])
    sources = dict(initial['sources'])
    import trackastra, ctc_metrics
    for package in [trackastra, ctc_metrics]:
        for p in Path(package.__file__).parent.rglob('*.py'):
            sources[str(p)] = sha(p)
    put(ROOT / 'EVALUATION_PREPARED.json', dict(time=time.time(), sources=sources, inputs=inputs,
        views=views, alpha=.25, graphs=90, evaluations=540,
        model_barrier_sha256=sha(ROOT / 'ALL_50_MODELS_BOUND.json'),
        source_seal_sha256=sha(ROOT / 'EVALUATION_SOURCE_SEALED.json'),
        scope='Post-hoc historically exposed video01; fixed native-trained parent heads; all 18 views; no reselection'))


def features(ds, gap, offset):
    plan = check()
    v = view_spec(ds, gap, offset)
    bind_read(v['images'] + v['masks'] + [str(encoder_path(ds))], plan)
    d = ROOT / 'features' / view_key(ds, gap, offset)
    assert not d.exists()
    d.mkdir(parents=True)
    torch.set_num_threads(1)
    arr = observable_features(v['images'], v['masks'], Encoder(encoder_path(ds)), node_geometry, crop_masks)
    save(d / 'features.npz', **arr)
    put(d / 'DONE.json', dict(time=time.time(), dataset=ds, gap=gap, offset=offset,
        source_frames=v['source_frames'], nodes=len(arr['t']),
        features_sha256=sha(d / 'features.npz'), encoder_sha256=sha(encoder_path(ds)),
        prepared_sha256=sha(ROOT / 'EVALUATION_PREPARED.json')))
    print('FEATURES', ds, gap, offset, len(arr['t']), flush=True)


def predict_all():
    from trackastra.model import Trackastra
    from trackastra.tracking import build_graph
    plan = check()
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    assert torch.cuda.device_count() == 1
    for ds in DATASETS:
        for seed in range(1, 6):
            model_dir = tracker_path(ds, seed)
            bind_read([str(model_dir / n) for n in ['model.pt', 'config.yaml', 'train_config.yaml']], plan)
            model = Trackastra.from_folder(model_dir, device='cuda')
            for gap, offset in [(1, 0)] + [(8, o) for o in range(8)]:
                v = view_spec(ds, gap, offset)
                bind_read(v['images'] + v['masks'], plan)
                d = ROOT / 'graphs' / view_key(ds, gap, offset) / f's{seed}'
                assert not d.exists()
                d.mkdir(parents=True)
                begin = time.time()
                imgs = np.stack([tifffile.imread(p) for p in v['images']])
                masks = np.stack([tifffile.imread(p) for p in v['masks']]).astype(np.int32)
                pred = model._predict(imgs, masks, edge_threshold=.05, n_workers=0, batch_size=1)
                graph = build_graph(pred['nodes'], pred['weights'], max_distance=256, max_neighbors=10, delta_t=1)
                areas = [np.bincount(m.ravel()) for m in masks]
                for n, attr in graph.nodes(data=True):
                    attr['area'] = int(areas[attr['time']][attr['label']])
                dump_pickle(d / 'graph.pkl', graph)
                put(d / 'DONE.json', dict(time=time.time(), started_unix=begin, dataset=ds, seed=seed,
                    gap=gap, offset=offset, nodes=len(graph), edges=graph.number_of_edges(),
                    graph_sha256=sha(d / 'graph.pkl'), source_frames=v['source_frames'],
                    model_sha256=sha(model_dir / 'model.pt'), prepared_sha256=sha(ROOT / 'EVALUATION_PREPARED.json'),
                    cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    supplied_detections=int(sum(len(np.unique(m))-1 for m in masks))))
                print('GRAPH', ds, seed, gap, offset, len(graph), flush=True)
                del imgs, masks, pred, graph
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()


def make_gt(ds, gap, offset):
    plan = check()
    v = view_spec(ds, gap, offset)
    bind_read(v['gt_masks'] + [v['track_file']], plan)
    rows = project_tracks(np.loadtxt(v['track_file'], dtype=np.int64, ndmin=2), v['source_frames'])
    d = ROOT / 'gt' / view_key(ds, gap, offset) / 'GT/TRA'
    d.mkdir(parents=True, exist_ok=False)
    (d / 'man_track.txt').write_text(''.join(' '.join(map(str, row))+'\n' for row in rows))
    ids = set(rows[:, 0]) | {0}
    for t, p in enumerate(v['gt_masks']):
        a = tifffile.imread(p)
        assert set(np.unique(a)) <= ids, 'Reference pixels missing from projected tracks'
        tifffile.imwrite(d / f'man_track{t:03d}.tif', a, compression='zlib')
    children = collections.Counter(int(p) for p in rows[:, 3] if p)
    put(d.parent / 'DONE.json', dict(time=time.time(), source_frames=v['source_frames'],
        files={p.name: sha(p) for p in d.iterdir()}, tracks=len(rows),
        observed_child_count_histogram=dict(collections.Counter(children.values())),
        parents_with_at_least_two_observed_children=sum(n >= 2 for n in children.values())))


def evaluate(ds, gap, offset, seed):
    from ctc_metrics import evaluate_sequence
    plan = check()
    torch.set_num_threads(1)
    v = view_spec(ds, gap, offset)
    key = view_key(ds, gap, offset)
    gd = ROOT / 'graphs' / key / f's{seed}'
    gr = read(gd / 'DONE.json')
    assert sha(gd / 'graph.pkl') == gr['graph_sha256']
    assert gr['prepared_sha256'] == sha(ROOT / 'EVALUATION_PREPARED.json')
    graph = load_pickle(gd / 'graph.pkl')
    fd = ROOT / 'features' / key
    fr = read(fd / 'DONE.json')
    assert sha(fd / 'features.npz') == fr['features_sha256']
    with np.load(fd / 'features.npz') as z:
        feature = {k: z[k].copy() for k in z.files}
    with np.load(TRAIN[ds] / 'scaling.npz') as z:
        scale = {k: z[k].copy() for k in z.files}
    bind_read(v['masks'], plan)
    masks = np.stack([tifffile.imread(p) for p in v['masks']]).astype(np.int32)
    gt = ROOT / 'gt' / key / 'GT'
    ref, ref_solver = capacity.decode(graph, 3)
    for arm in ARMS:
        d = ROOT / 'results' / key / f'{arm}_s{seed}'
        d.mkdir(parents=True, exist_ok=False)
        begin = time.time()
        if arm == 'reference':
            solution, solver, model_hash = ref, ref_solver, None
        else:
            model_path = checkpoint(ds, arm, seed)
            bind_read([str(model_path)], plan)
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            assert ckpt['arm'] == arm and ckpt['seed'] == seed and ckpt['updates'] == 6000
            assert ckpt['training_seal_sha256'] == sha(TRAIN[ds] / 'TRAINING_SEALED.json')
            model = ParentModel()
            model.load_state_dict(ckpt['state_dict'], strict=True)
            model.eval()
            scores = score_graph(graph, model, arm, feature, scale, pair_geometry, neutral_times=v['neutral_times'])
            save(d / 'scores.npz', **scores)
            evidence = {(int(p), int(c)): float(r) for p, c, r in
                        zip(scores['parent_node'], scores['child_node'], scores['residual'])}
            solution, solver = association_decoder.decode(graph, ref, evidence, .25)
            model_hash = sha(model_path)
        dump_pickle(d / 'solution.pkl', solution)
        put(d / 'SOLVER.json', solver)
        export(solution, masks, d / 'RES')
        metrics = clean_metrics(evaluate_sequence(str(d / 'RES'), str(gt),
            metrics=['TRA', 'DET', 'LNK', 'BC', 'CT', 'CHOTA'], threads=1))
        assert all(metrics[k] is not None for k in METRICS)
        if arm == 'reference':
            gt_divisions = metrics['gt_divisions']
        assert metrics['gt_divisions'] == gt_divisions
        put(d / 'RESULT.json', dict(dataset=ds, gap=gap, offset=offset, seed=seed, arm=arm,
            alpha=0 if arm == 'reference' else .25, metrics=metrics, started_unix=begin, finished_unix=time.time(),
            source_frames=v['source_frames'], prepared_sha256=sha(ROOT / 'EVALUATION_PREPARED.json'),
            graph_sha256=gr['graph_sha256'], features_sha256=fr['features_sha256'],
            model_sha256=model_hash, score_sha256=None if arm == 'reference' else sha(d / 'scores.npz'),
            solution_sha256=sha(d / 'solution.pkl'), solver_sha256=sha(d / 'SOLVER.json'),
            export_hashes={p.name: sha(p) for p in (d / 'RES').iterdir()},
            out_degree_counts=dict(collections.Counter(solution.out_degree(n) for n in solution))))
        print('EVALUATED', key, seed, arm, metrics['BC(0)'], metrics['CT'], flush=True)


def cpu_jobs(mode, jobs, workers=3):
    pending, active, failed = list(jobs), [], []
    while pending or active:
        while pending and len(active) < workers and not failed:
            args = pending.pop(0)
            name = '_'.join(map(str, args))
            logdir = ROOT / 'logs'
            logdir.mkdir(exist_ok=True)
            log = (logdir / f'{mode}_{name}.log').open('x')
            cmd = [sys.executable, '-u', str(HERE / 'run.py'), mode, '--dataset', args[0],
                   '--gap', str(args[1]), '--offset', str(args[2])]
            if len(args) == 4:
                cmd += ['--seed', str(args[3])]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            active.append((p, log, name))
        time.sleep(3)
        for p, log, name in list(active):
            if p.poll() is not None:
                log.close()
                active.remove((p, log, name))
                if p.returncode:
                    failed.append((name, p.returncode))
        if failed and not active:
            break
    assert not failed, failed


def wait_for_models():
    launch = read(ROOT / 'START_REQUEST.json')
    while not (TRAIN['fluo'] / 'ALL_MODELS_TRAINED.json').exists():
        proc = Path('/proc') / str(launch['pid'])
        assert proc.exists(), 'Training process ended without completed models; no automatic restart'
        status = (proc / 'stat').read_text().rsplit(')', 1)[1].strip().split()[0]
        cmd = (proc / 'cmdline').read_bytes()
        assert status != 'Z' and str(ROOT / 'source/pipeline.py').encode() in cmd
        assert not list(TRAIN['fluo'].glob('*FAILURE*.json'))
        time.sleep(20)


def idle_gpu():
    """Choose by resource availability, never by scores; do not evict any process."""
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    occupied = set(used.splitlines())
    for line in text.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(',')]
        if uuid not in occupied and int(memory) < 100 and int(utilization) == 0:
            return index
    return None


def driver():
    lock = (ROOT / 'EVALUATION.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    put(ROOT / 'EVALUATION_STARTED.json', dict(time=time.time(), pid=os.getpid(), waiting_for_fluo_models=True))
    wait_for_models()
    prepare()
    jobs = [(ds, gap, o) for ds in DATASETS for gap, o in [(1, 0)] + [(8, k) for k in range(8)]]
    cpu_jobs('features', jobs)
    for args in jobs:
        make_gt(*args)
    gpu = idle_gpu()
    while gpu is None:
        time.sleep(30)
        gpu = idle_gpu()
    with (ROOT / 'predict.log').open('x') as log:
        p = subprocess.run([sys.executable, '-u', str(HERE / 'run.py'), 'predict'],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
    assert p.returncode == 0
    assert len(list((ROOT / 'graphs').glob('*/*/DONE.json'))) == 90
    put(ROOT / 'ALL_GRAPHS_DONE.json', dict(time=time.time(), graphs=90,
        bindings={str(p): sha(p) for p in (ROOT / 'graphs').glob('*/*/DONE.json')}))
    cpu_jobs('evaluate', [(*j, s) for j in jobs for s in range(1, 6)])
    rows = [read(p) for p in sorted((ROOT / 'results').glob('*/*/RESULT.json'))]
    assert len(rows) == 540
    put(ROOT / 'SUMMARY.json', dict(time=time.time(), groups=aggregate(rows, METRICS), evaluations=540,
        scope=read(ROOT / 'EVALUATION_PREPARED.json')['scope']))
    put(ROOT / 'SCORED.json', dict(time=time.time(), status='PENDING_INDEPENDENT_VERIFICATION',
                                 summary_sha256=sha(ROOT / 'SUMMARY.json')))
    with (ROOT / 'verification.log').open('x') as log:
        p = subprocess.run([sys.executable, '-u', str(HERE / 'verify.py')], stdout=log, stderr=subprocess.STDOUT)
    assert p.returncode == 0 and read(ROOT / 'VERIFIED.json')['status'] == 'VERIFIED'
    put(ROOT / 'DONE.json', dict(time=time.time(), status='VERIFIED',
                               verification_sha256=sha(ROOT / 'VERIFIED.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['driver', 'features', 'predict', 'evaluate'])
    parser.add_argument('--dataset', choices=list(DATASETS))
    parser.add_argument('--gap', type=int)
    parser.add_argument('--offset', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'driver':
            driver()
        elif args.mode == 'predict':
            predict_all()
        elif args.mode == 'features':
            features(args.dataset, args.gap, args.offset)
        else:
            evaluate(args.dataset, args.gap, args.offset, args.seed)
    except BaseException:
        import traceback
        put(ROOT / f'EVALUATION_FAILURE_{os.getpid()}_{time.time_ns()}.json',
            dict(time=time.time(), traceback=traceback.format_exc()))
        raise
