"""Frozen Fluo-source transfer to the complete, named 53-field LFCT cohort.

No target training or model selection. Shared image-only Cellpose-SAM detections;
source-only native/sparse policies. Ground-truth processing is confined to the
evaluation worker. Large inputs/weights stay on the server.
"""
from pathlib import Path
import collections, concurrent.futures, fcntl, hashlib, importlib.util, json, os, pickle, subprocess, sys, time
import numpy as np

N=Path('STUDY_ROOT');E=N/'eval/mitosync'
ROOT=E/'reviewer_sprint_20261101/lfct_v1'
UP=Path('EXTERNAL_DATA_ROOT')
ST=E/'division_triplet_20261025/v1';MR=E/'multirate_parent_20261025/v2';DP=E/'direct_paint_20261101/v1';AB=E/'parent_ablation_20261029/v1'
FULL=E/'parent_fullview_20261024/v1';HERE=Path(__file__).resolve().parent
PY='EXTERNAL_DATA_ROOT'
ARMS=['R','RG','ProgramPaintMatched','RenderedMatched'];VIEWS=[(1,0)]+[(8,o) for o in range(8)]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def progress(p,v):
    q=Path(str(p)+f'.{os.getpid()}.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def check(d):
    for p,h in d.items():assert sha(p)==h,p
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def modelpath(arm,s):return (ST if arm in ['R','RG'] else DP)/'models'/f'fluo_{arm}_s{s}/model.pt'
def tracker(s,gap=1):return E/'runs'/f"Fluo-N2DL-HeLa_{'ChHead' if gap==1 else 'R18'}_s{s}"
def viewdir(case,gap,o):return ROOT/'views'/case/f'g{gap}_o{o}'
def lock(p):
    p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f

def seal():
    assert not (ROOT/'SEALED_v2.json').exists()
    cohort=read(UP/'LFCT_cohort.json');assert len(cohort)==53 and len({x['case'] for x in cohort})==53
    bindings={str(UP/'LFCT_cohort.json'):sha(UP/'LFCT_cohort.json')}
    assert read(DP/'INDEPENDENT_READBACK.json')['status']=='PASS'
    for p in [DP/'INDEPENDENT_READBACK.json',DP/'ALL_MODELS_TRAINED.json',ST/'ALL_MODELS_TRAINED.json',MR/'TRAINING_SEALED_v2.json',E/'fluo_native_selection_20261025/v1/SELECTED.json']:
        bindings[str(p)]=sha(p)
    models={}
    for s in range(1,6):
        for arm in ARMS:
            p=modelpath(arm,s);models[f'{arm}_s{s}']=dict(path=str(p),sha256=sha(p));bindings[str(p)]=sha(p)
        for n in ['model.pt','config.yaml','train_config.yaml']:
            p=tracker(s)/n;bindings[str(p)]=sha(p)
    enc=E/'pixel_div_20261018/runs/fluo_R_s1/model.pt';bindings[str(enc)]=sha(enc)
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    for folder in [MR/'source',ST/'source',ST/'evaluation_source',AB/'source',FULL/'evaluation_source']:
        for p in folder.glob('*.py'):sources[str(p)]=sha(p)
    import trackastra,cellpose
    for package in [trackastra,cellpose]:
        for p in Path(package.__file__).parent.rglob('*.py'):sources[str(p)]=sha(p)
    from cellpose.models import MODEL_DIR
    cp=Path(MODEL_DIR)/'cpsam';assert cp.exists();bindings[str(cp)]=sha(cp)
    put(ROOT/'SEALED_v2.json',dict(time=time.time(),cohort=cohort,models=models,bindings=bindings,sources=sources,
        encoder=str(enc),source_dataset='Fluo-N2DL-HeLa',target_channel='RFP',seeds=[1,2,3,4,5],views=VIEWS,
        primary_contrasts=['RG minus R','RenderedMatched minus ProgramPaintMatched'],
        decoder=dict(alpha=0,native_beta=.5,sparse_beta=.25,cap=3,source_policy='Final fixed v63 policy; no LFCT feedback'),
        eligibility='All 53 named fields requested. Missing/corrupt sources reported. GT multiple-parent cases held for annotation adjudication; no parent selected automatically. Out-of-bounds points retained as unmatched.',
        metrics='Exact observed-frame binary mother/two-daughter match via one-to-one centroid-in-mask assignment. All predicted branches enter precision. Nonbinary observed GT branches reported separately. This is not CTC mask-IoU BC(0).',
        spatial_convention='Float coordinates in image, nearest pixel floor(x+.5); no clipping or nearest-cell substitution.',
        sparse_convention='Follow unique GT ancestry between consecutive retained frames, retaining two-child observed divisions. Channels, offsets and seeds are not extra independent fields.',
        reporting='All fields and failures. Native primary, eight-offset average secondary; field macro for GT-positive fields and pooled counts including zero-event fields. Average seeds before field comparisons. Fields are not proven independent biological replicates.',
        prior_use='MitoSync checkpoints frozen before this study. LFCT is concurrently used in separate CellTopo work; no CellTopo model scores or outputs are used. General pretraining-data exclusion is not established.',
        input_reuse='Reuse complete image-only Cellpose masks after hash binding; otherwise generate in this study directory with identical frozen settings. Never write to CellTopo inputs/results.',
        deadline_unix=time.time()+11*3600))

def command(log,args,env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:subprocess.run([PY,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)

def prepare_case(rec):
    case=rec['case'];w=ROOT/'cases'/case
    if (w/'ADMISSION.json').exists() or (w/'PREPARE_FAILURE.json').exists():return
    original=UP/'cases'/case
    if not (original/'input_ready.json').exists():return
    w.mkdir(parents=True,exist_ok=True)
    r=read(original/'input_ready.json');assert r['case']==case
    # Copy complete annotation metadata; shared pixels are never copied or edited.
    gt=read(original/'ground_truth.json');assert gt['source_sha256']==r['annotation_sha256']
    assert sha(UP/'raw'/case/'annotations.xml')==r['annotation_sha256']
    assert len(r['frame_manifest'])==r['frames']
    images=[];bindings={}
    for f in r['frame_manifest']:
        p=original/'img'/f"t{f['frame']:04d}.tif";assert sha(p)==f['sha256'];images.append(str(p));bindings[str(p)]=f['sha256']
    assert len(images)==r['frames'] and [x['frame'] for x in r['frame_manifest']]==list(range(r['frames']))
    put(w/'ground_truth.json',gt);put(w/'input_ready.json',r)
    (w/'img').symlink_to(original/'img',target_is_directory=True)
    merges=[x for x in r.get('annotation_review_required',[]) if x['kind']=='multiple_parents']
    put(w/'ADMISSION.json',dict(case=case,time=time.time(),images=images,bindings=bindings,
        ground_truth_sha256=sha(w/'ground_truth.json'),annotation_sha256=r['annotation_sha256'],
        original_ready_sha256=sha(original/'input_ready.json'),held=bool(merges),
        reason='Original annotation contains multiple parents; preserve and hold all methods' if merges else None,
        outside_points_retained=sum(x['kind']=='out_of_bounds_centroid' for x in r.get('annotation_review_required',[])),
        independent_of_model_scores=True))

def masks(case,gpu):
    w=ROOT/'cases'/case;r=read(w/'input_ready.json');old=UP/'cases'/case;dest=w/'MASKS_READY.json'
    if dest.exists():return read(dest)
    f=lock(w/'MASKS.lock')
    if f is None:return None
    try:
        if dest.exists():return read(dest)
        candidates=sorted((old/'det/cpsam').glob('det*.tif'))
        meta=old/'det/cpsam/meta.json'
        borrow=meta.exists() and len(candidates)==r['frames']
        if borrow:
            m=read(meta);assert m['model']=='cpsam' and m['diameter'] is None and m['flow_threshold']==.4 and m['cellprob_threshold']==0
        else:
            # Each study owns its output directory; never race the other queue.
            # A v1 segmentation subprocess may finish while its scheduler is held.
            # Recover only a complete, validated mask boundary; retain its original log.
            meta=w/'det/cpsam/meta.json';candidates=sorted((w/'det/cpsam').glob('det*.tif'))
            recovered=meta.exists() and len(candidates)==r['frames']
            if recovered:
                m=read(meta);assert m['model']=='cpsam' and m['diameter'] is None and m['flow_threshold']==.4 and m['cellprob_threshold']==0
                put(w/'SEGMENTATION_BOUNDARY_RECOVERY.json',dict(time=time.time(),reason='Completed v1 child preserved across candidate-checkpoint wiring repair',meta_sha256=sha(meta)))
            else:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                command(w/'segmentation.log',[HERE/'run_cpsam_frozen.py',w,'--gpu',gpu,'--batch',8],env)
                candidates=sorted((w/'det/cpsam').glob('det*.tif'));meta=w/'det/cpsam/meta.json'
        assert [p.name for p in candidates]==[f'det{i:04d}.tif' for i in range(r['frames'])]
        binding={str(p):sha(p) for p in candidates};binding[str(meta)]=sha(meta)
        put(dest,dict(time=time.time(),paths=list(map(str,candidates)),bindings=binding,borrowed_complete_image_only_detections=borrow))
        return read(dest)
    finally:f.close()

def gpu_headroom(gpu):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    row=next(x.split(',') for x in rows if x.split(',')[0].strip()==str(gpu));uuid=row[1].strip()
    if int(row[2])<20000:return False
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    for line in apps:
        if line.split(',')[0].strip()==uuid:
            p=Path('/proc')/line.split(',')[1].strip()
            if p.exists() and p.stat().st_uid!=os.getuid():return False
    return True

def gpu_view(case,gap,offset,gpu):
    import torch,tifffile
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.12)
    plan=read(ROOT/'SEALED_v2.json');w=ROOT/'cases'/case;a=read(w/'ADMISSION.json');assert not a['held']
    mr=read(w/'MASKS_READY.json');check(a['bindings']);check(mr['bindings'])
    kept=list(range(offset,len(a['images']),gap));d=viewdir(case,gap,offset);d.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(MR/'source'));from encoder import Encoder
    F=module('external_observed_features',MR/'source/features.py')
    imgs=[tifffile.imread(a['images'][t]) for t in kept];ms=[tifffile.imread(mr['paths'][t]).astype(np.int32) for t in kept]
    assert all(x.shape==ms[0].shape for x in imgs+ms)
    enc=Encoder(plan['encoder']).to('cuda:0');cpu=Encoder(plan['encoder'])
    features,probes=F.view_features([F.norm_u8(x) for x in imgs],ms,enc,gap,'cuda:0')
    pp=[probes[k] for k in sorted({0,len(probes)//2,len(probes)-1})]
    with torch.inference_mode():
        z=cpu(torch.from_numpy(np.stack([x['crop'] for x in pp])),torch.from_numpy(np.stack([x['mask'] for x in pp]))).numpy().astype(np.float16)
    np.testing.assert_allclose(z,np.stack([x['embedding'] for x in pp]),atol=.0003,rtol=.003)
    with (d/'features.npz').open('xb') as f:np.savez_compressed(f,**features)
    put(d/'FEATURES_DONE.json',dict(time=time.time(),sha256=sha(d/'features.npz'),kept=kept,nodes=len(features['t']),
        input_admission_sha256=sha(w/'ADMISSION.json'),masks_ready_sha256=sha(w/'MASKS_READY.json'),encoder_sha256=sha(plan['encoder']),cpu_probes=len(pp)))
    del enc,cpu;torch.cuda.empty_cache()
    from trackastra.model import Trackastra
    from trackastra.tracking import build_graph
    image_array=np.stack(imgs);mask_array=np.stack(ms);areas=[np.bincount(m.ravel()) for m in ms]
    for seed in range(1,6):
        model=Trackastra.from_folder(tracker(seed,gap),device='cuda')
        pred=model._predict(image_array,mask_array,edge_threshold=.05,n_workers=0,batch_size=1)
        graph=build_graph(pred['nodes'],pred['weights'],max_distance=256,max_neighbors=10,delta_t=1)
        for n,v in graph.nodes(data=True):v['area']=int(areas[v['time']][v['label']])
        assert {(int(v['time']),int(v['label'])) for n,v in graph.nodes(data=True)}==set(zip(map(int,features['t']),map(int,features['seg_id'])))
        gd=d/f's{seed}';gd.mkdir(exist_ok=False)
        with (gd/'graph.pkl').open('xb') as f:pickle.dump(graph,f,protocol=4)
        put(gd/'GRAPH_DONE.json',dict(time=time.time(),seed=seed,sha256=sha(gd/'graph.pkl'),nodes=len(graph),edges=graph.number_of_edges(),
            model_sha256=sha(tracker(seed,gap)/'model.pt'),features_receipt_sha256=sha(d/'FEATURES_DONE.json')))
        del model,pred,graph;torch.cuda.empty_cache()
    check(a['bindings']);check(mr['bindings'])
    put(d/'PREDICTION_DONE.json',dict(time=time.time(),seed_receipts={str(p):sha(p) for p in d.glob('s*/GRAPH_DONE.json')}))

def gpu_worker(shard,gpu):
    plan=read(ROOT/'SEALED_v2.json');check(plan['bindings']);check(plan['sources'])
    while time.time()<plan['deadline_unix']:
        progressed=False
        if not gpu_headroom(gpu):time.sleep(30);continue
        for gap,offset in VIEWS:
            for rec in plan['cohort']:
                case=rec['case'];w=ROOT/'cases'/case;a=w/'ADMISSION.json'
                if not a.exists() or read(a)['held'] or (w/'MASK_FAILURE.json').exists():continue
                d=viewdir(case,gap,offset)
                if (d/'PREDICTION_DONE.json').exists() or (d/'FAILURE.json').exists():continue
                f=lock(d/'PREDICT.lock')
                if f is None:continue
                try:
                    if (d/'PREDICTION_DONE.json').exists() or (d/'FAILURE.json').exists():continue
                    mask=masks(case,gpu)
                    if mask is None:continue
                    progressed=True
                    progress(ROOT/f'GPU_{shard}_STATUS.json',dict(time=time.time(),case=case,gap=gap,offset=offset,stage='predicting'))
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                    command(d/'prediction.log',[HERE/'lfct_external.py','view',case,gap,offset,gpu],env)
                except Exception:
                    import traceback
                    put(d/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
                finally:f.close()
        all_admitted=all((ROOT/'cases'/x['case']/'ADMISSION.json').exists() for x in plan['cohort'])
        if not progressed and all_admitted:break
        if not progressed:time.sleep(30)
    progress(ROOT/f'GPU_{shard}_STATUS.json',dict(time=time.time(),stage='queue_complete_or_deadline'))

def metric_mapping(case,gap,offset):
    import tifffile
    from lfct_metrics import prepare_mapping
    w=ROOT/'cases'/case;d=viewdir(case,gap,offset);p=d/'GT_MAPPING.json'
    if p.exists():return read(p)
    while True:
        f=lock(d/'GT_MAPPING.lock')
        if f is not None:break
        time.sleep(1)
        if p.exists():return read(p)
    try:
        if p.exists():return read(p)
        admission=read(w/'ADMISSION.json');assert not admission['held'] and sha(w/'ground_truth.json')==admission['ground_truth_sha256']
        fr=read(d/'FEATURES_DONE.json');mr=read(w/'MASKS_READY.json');check(mr['bindings'])
        kept=fr['kept'];gt=read(w/'ground_truth.json');pr,mp,audit=prepare_mapping(gt,kept,[tifffile.imread(mr['paths'][i]) for i in kept])
        put(p,dict(projected=pr,mapping=mp,audit=audit,time=time.time(),gt_sha256=sha(w/'ground_truth.json'),mask_binding_sha256=sha(w/'MASKS_READY.json')))
        return read(p)
    finally:f.close()

def evaluate(case,gap,offset,seed):
    import torch
    torch.set_num_threads(1)
    from lfct_metrics import counts
    d=viewdir(case,gap,offset);gd=d/f's{seed}';r=read(gd/'GRAPH_DONE.json');assert sha(gd/'graph.pkl')==r['sha256']
    with (gd/'graph.pkl').open('rb') as f:g=pickle.load(f)
    fr=read(d/'FEATURES_DONE.json');assert sha(d/'features.npz')==fr['sha256']
    with np.load(d/'features.npz') as z:feat={k:z[k] for k in z.files}
    mm=metric_mapping(case,gap,offset);mapping={k:tuple(v) for k,v in mm['mapping'].items()}
    sys.path.insert(0,str(ST/'source'))
    model=module('lfct_pair_model',ST/'source/model.py');score=module('lfct_pair_score',ST/'evaluation_source/inference.py')
    decoder=module('lfct_pair_decoder',AB/'source/decoder.py');cap=module('lfct_graph_capacity',FULL/'evaluation_source/capacity.py')
    reference,rr=cap.decode(g,3);rows={};hashes={};beta=.5 if gap==1 else .25
    from trackastra.tracking import track_greedy
    greedy=track_greedy(g.copy());greedy.add_nodes_from(g.nodes(data=True))
    for arm in ['TrackastraGraph','TrackastraGreedy']+ARMS:
        out=gd/arm;out.mkdir()
        if arm=='TrackastraGraph':sol,sr=reference,rr
        elif arm=='TrackastraGreedy':sol,sr=greedy,dict(native_linker=True)
        else:
            cp=modelpath(arm,seed);ck=torch.load(cp,map_location='cpu',weights_only=True)
            assert ck['seed']==seed and ck['updates']==6000
            net=model.TripletModel();net.load_state_dict(ck['state_dict'],strict=True);net.eval()
            sc=score.score_graph(g,net,'RG',feat)
            with (out/'triplets.npz').open('xb') as f:np.savez_compressed(f,**sc)
            pairs={(int(p),int(a),int(b)):float(v) for p,a,b,v in zip(sc['mother'],sc['daughter_a'],sc['daughter_b'],sc['residual'])}
            sol,sr=decoder.decode(g,reference,{e:0. for e in g.edges},pairs,beta,0.)
            assert abs(sr['objective']-decoder.objective(g,list(sol.edges),{e:0. for e in g.edges},0.,pairs,beta))<1e-7
        metrics=counts(sol,mm['projected'],mapping)
        # Graph+fixed input masks suffice to reproduce every lineage pixel export.
        with (out/'solution.pkl').open('xb') as f:pickle.dump(sol,f,protocol=4)
        put(out/'SOLVER.json',sr)
        row=dict(case=case,gap=gap,offset=offset,seed=seed,arm=arm,metrics=metrics,
            input_mapping_sha256=sha(d/'GT_MAPPING.json'),graph_sha256=sha(gd/'graph.pkl'),features_sha256=sha(d/'features.npz'),
            model_sha256=None if arm.startswith('Trackastra') else sha(modelpath(arm,seed)),
            solution_sha256=sha(out/'solution.pkl'),solver_sha256=sha(out/'SOLVER.json'),time=time.time())
        put(out/'RESULT.json',row);rows[arm]=metrics;hashes[str(out/'RESULT.json')]=sha(out/'RESULT.json')
    put(gd/'EVALUATION_DONE.json',dict(time=time.time(),results=hashes,metric='LFCT centroid-in-mask, not CTC BC(0)'))

def evaluation_driver():
    plan=read(ROOT/'SEALED_v2.json');check(plan['bindings']);check(plan['sources'])
    active={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        while time.time()<plan['deadline_unix']+1800:
            for future in list(active):
                if future.done():
                    gd=active.pop(future)
                    try:future.result()
                    except Exception:
                        import traceback
                        put(gd/'EVALUATION_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
            busy=set(active.values())
            for gap,o in VIEWS:
                for rec in plan['cohort']:
                    for seed in range(1,6):
                        gd=viewdir(rec['case'],gap,o)/f's{seed}'
                        if len(active)>=10:break
                        if not (gd/'GRAPH_DONE.json').exists() or gd in busy or (gd/'EVALUATION_DONE.json').exists() or (gd/'EVALUATION_FAILURE.json').exists():continue
                        f=pool.submit(command,gd/'evaluation.log',[HERE/'lfct_external.py','eval',rec['case'],gap,o,seed])
                        active[f]=gd
            progress(ROOT/'EVALUATION_STATUS.json',dict(time=time.time(),active=len(active),complete=len(list((ROOT/'views').glob('*/*/s*/EVALUATION_DONE.json'))),failures=len(list((ROOT/'views').glob('*/*/s*/EVALUATION_FAILURE.json')))))
            if (ROOT/'PREDICTION_QUEUE_DONE.json').exists() and not active:break
            time.sleep(15)

def driver():
    f=lock(ROOT/'DRIVER.lock');assert f is not None
    command(ROOT/'unit_tests_v2.log',[HERE/'test_lfct_metrics.py','-v'])
    plan=read(ROOT/'SEALED_v2.json');check(plan['bindings']);check(plan['sources']);workers=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for mode,args in [('gpu',['0','2']),('gpu',['1','2']),('evaluate_queue',[])]:
        log=(ROOT/('_'.join([mode,*args])+'_v2.log')).open('x')
        p=subprocess.Popen([PY,'-u',str(HERE/'lfct_external.py'),mode,*args],env=env,stdout=log,stderr=subprocess.STDOUT);workers.append((p,log,mode))
    put(ROOT/'WORKERS_V2.json',dict(time=time.time(),workers=[dict(pid=p.pid,role=m) for p,l,m in workers]))
    while time.time()<plan['deadline_unix']:
        for rec in plan['cohort']:
            try:prepare_case(rec)
            except Exception:
                import traceback
                put(ROOT/'cases'/rec['case']/'PREPARE_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        admission=list((ROOT/'cases').glob('*/ADMISSION.json'))
        progress(ROOT/'INPUT_STATUS.json',dict(time=time.time(),admitted=len(admission),held=sum(read(p)['held'] for p in admission),requested=53))
        if all(p.poll() is not None for p,l,m in workers if m=='gpu'):break
        time.sleep(20)
    for p,l,m in workers:
        if m=='gpu':p.wait();l.close()
    put(ROOT/'PREDICTION_QUEUE_DONE.json',dict(time=time.time(),gpu_workers=[dict(pid=p.pid,returncode=p.returncode) for p,l,m in workers if m=='gpu']))
    for p,l,m in workers:
        if m=='evaluate_queue':p.wait();l.close()
    put(ROOT/'QUEUE_DONE.json',dict(time=time.time(),workers=[dict(pid=p.pid,role=m,returncode=p.returncode) for p,l,m in workers]))

if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode=='driver':driver()
        elif mode=='gpu':gpu_worker(int(sys.argv[2]),int(sys.argv[3]))
        elif mode=='view':gpu_view(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
        elif mode=='eval':evaluate(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
        elif mode=='evaluate_queue':evaluation_driver()
        else:raise ValueError(mode)
    except BaseException:
        import traceback
        put(ROOT/f'FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
