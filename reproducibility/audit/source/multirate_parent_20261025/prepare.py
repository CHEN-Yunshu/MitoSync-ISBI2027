"""Prepare all observed-rate training views; no development/test pixels."""
import collections,fcntl,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from common import *

def check():
    seal=read(ROOT/'SEALED.json')
    for p,h in {**seal['sources'],**seal['inputs']}.items():assert sha(p)==h,p
    return seal

def device():
    try:
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
        users=collections.defaultdict(list)
        for line in processes:
            uuid,pid=line.split(',');users[uuid.strip()].append(int(pid))
        for line in gpus:
            index,uuid,free=[x.strip() for x in line.split(',')]
            if not users[uuid] and int(free)>16000:return dict(gpu=int(index),mode='idle',pids=[],free_mib=int(free))
        for line in gpus:
            index,uuid,free=[x.strip() for x in line.split(',')]
            if index!='0' or int(free)<40960:continue
            commands=[]
            for pid in users[uuid]:
                p=Path('/proc')/str(pid);assert p.stat().st_uid==os.getuid()
                cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                assert str(E/'scf_baseline_20261024/v1/full_training_source_v1/full_training.py') in cmd
                commands.append(cmd)
            return dict(gpu=0,mode='share_own_SCF_only',pids=users[uuid],commands=commands,free_mib=int(free))
    except (subprocess.CalledProcessError,FileNotFoundError,AssertionError):pass
    return dict(gpu=None,mode='cpu')

def seal():
    inputs={};jobs=[];specs={}
    for ds,sp in SPECS.items():
        s=read(sp);old=read(OLD[ds]/'PRETRAIN_VERIFIED.json')
        assert old['status']=='VERIFIED' and old['prepared_sha256']==sha(OLD[ds]/'PREPARED.json')
        for p in [sp,ENCODERS[ds],OLD[ds]/'PREPARED.json',OLD[ds]/'PRETRAIN_VERIFIED.json']:inputs[str(p)]=sha(p)
        specs[ds]={}
        for suffix in SUFFIXES:
            source=ds+'_'+suffix;vs=s['sources'][source]['videos'];specs[ds][source]=vs
            assert len(vs)==(1 if suffix=='head' else (191 if ds=='psc' else 200))
            for i,v in enumerate(vs):
                if suffix=='head':assert '_trainhead/02' in v['img_dir']
                assert '_valtail' not in v['img_dir'] and '/01' not in v['name']
                assert v['T']>=16
                jobs.append(dict(ds=ds,source=source,index=i))
    assert len(jobs)==1175
    put(ROOT/'SOURCE_SPEC.json',specs)
    inputs[str(ROOT/'SOURCE_SPEC.json')]=sha(ROOT/'SOURCE_SPEC.json')
    put(ROOT/'SEALED.json',dict(time=time.time(),inputs=inputs,sources={str(p):sha(p) for p in HERE.glob('*.py')},
        protocol_sha256=sha(HERE/'PROTOCOL.md'),jobs=jobs,gaps=list(GAPS),all_offsets=True,
        context='previous/current/next observed; replicate endpoint; explicit flags',node_dim=6,edge_dim=15))

def independent_targets(a,rows,tracks):
    children=rows['child'];parents=rows['parents'];ids=rows['track_id'];lookup={}
    for i,(t,k) in enumerate(zip(a['t'],ids)):
        if k>0:lookup.setdefault((int(t),int(k)),[]).append(i)
    offspring=collections.Counter(int(p) for _,_,p in tracks.values() if p)
    for j,c in enumerate(children):
        t=int(a['t'][c]);label=int(ids[c]);b,e,p=tracks[label]
        if b<t:expected=label;kind=0
        elif p==0:expected=0;kind=2
        else:
            assert tracks[p][1]==t-1;expected=p;kind=int(offspring[p]>=2)
        y=int(rows['target'][j]);assert int(rows['kind'][j])==kind
        if expected==0:assert y==0
        else:
            assert len(lookup[(t-1,expected)])==1 and y>0
            assert int(parents[j,y])==lookup[(t-1,expected)][0] and rows['valid'][j,y]
        assert parents[j,0]==-1 and rows['valid'][j,0]

def video_cache(job,gpu,encoder,cpu_encoder):
    import torch,tifffile,hashlib
    from features import view_features,targets_for_view,norm_u8
    ds,source,index=job['ds'],job['source'],job['index'];spec=read(ROOT/'SOURCE_SPEC.json')[ds][source][index]
    dest=ROOT/ds/'cache'/source/f'{index:03d}';dest.mkdir(parents=True,exist_ok=False)
    begin=time.time();inputs={};images=[];segs=[];tras=[];image_hashes={}
    for t in range(spec['T']):
        paths=[Path(spec['img_dir'])/f't{t:03d}.tif',Path(spec['seg_dir'])/f'man_seg{t:03d}.tif',Path(spec['tra_dir'])/f'man_track{t:03d}.tif']
        for p in paths:inputs[str(p)]=sha(p)
        image_hashes[paths[0].name]=inputs[str(paths[0])]
        images.append(norm_u8(tifffile.imread(paths[0])));segs.append(tifffile.imread(paths[1]).astype(np.int32));tras.append(tifffile.imread(paths[2]).astype(np.int32))
    trackfile=Path(spec['tra_dir'])/'man_track.txt';inputs[str(trackfile)]=sha(trackfile);tr=np.loadtxt(trackfile,dtype=np.int64,ndmin=2)
    arrays=collections.defaultdict(list);views=[];shift=0;maxerr=0.;probecount=0;total=collections.Counter()
    for gap in GAPS:
        for offset in range(gap):
            kept=list(range(offset,len(images),gap))
            if len(kept)<2:continue
            im=[images[t] for t in kept];sg=[segs[t] for t in kept];ta=[tras[t] for t in kept]
            a,probes=view_features(im,sg,encoder,gap,gpu)
            rows,audit,projected=targets_for_view(a,sg,ta,tr,kept)
            independent_targets(a,rows,{int(k):(int(b),int(e),int(p)) for k,b,e,p in projected})
            # Three predetermined cells per view, CPU recomputation from raw observed pixels.
            selected=[probes[i] for i in sorted({0,len(probes)//2,len(probes)-1})]
            pixels=[];masks=[];expected=[]
            for p in selected:
                t,y,x=p['t'],p['cy'],p['cx'];idx=[max(0,t-1),t,min(len(im)-1,t+1)]
                crop=np.stack([np.pad(im[k],24,mode='reflect')[y:y+48,x:x+48] for k in idx])
                mask=np.pad(sg[t],24,mode='constant')[y:y+48,x:x+48]==p['seg_id']
                np.testing.assert_array_equal(crop,p['crop']);np.testing.assert_array_equal(mask,p['mask'])
                pixels.append(crop);masks.append(mask);expected.append(p['embedding'])
            with torch.inference_mode():fresh=cpu_encoder(torch.from_numpy(np.stack(pixels)),torch.from_numpy(np.stack(masks))).numpy().astype(np.float16)
            expected=np.stack(expected);np.testing.assert_allclose(fresh,expected,atol=.0003,rtol=.003)
            maxerr=max(maxerr,float(abs(fresh.astype(float)-expected).max()));probecount+=len(expected)
            vid=len(views);n=len(a['t']);nr=len(rows['child'])
            for k in ['t','cy','cx','area','seg_id','embedding','node_geometry']:arrays[k].append(a[k])
            arrays['track_id'].append(rows['track_id']);arrays['view_id'].append(np.full(n,vid,np.int16))
            arrays['child'].append(rows['child']+shift);pp=rows['parents'].copy();pp[pp>=0]+=shift;arrays['parents'].append(pp)
            for k in ['valid','target','kind','edge_geometry']:arrays[k].append(rows[k])
            arrays['gap'].append(np.full(nr,gap,np.int8));arrays['row_view'].append(np.full(nr,vid,np.int16))
            views.append(dict(gap=gap,offset=offset,kept=kept,node_begin=shift,nodes=n,rows=nr,projected_tracks=projected.tolist(),audit=audit))
            shift+=n;total.update(audit)
    combined={k:np.concatenate(v) for k,v in arrays.items()};assert len(views)==15
    save(dest/'features.npz',**combined);put(dest/'VIEWS.json',views)
    h=hashlib.sha256()
    for k in sorted(combined):
        if k!='embedding':h.update(k.encode());h.update(combined[k].tobytes())
    h.update(__import__('json').dumps(views,sort_keys=True).encode())
    for p,digest in inputs.items():assert sha(p)==digest,p
    put(dest/'DONE.json',dict(time=time.time(),ds=ds,source=source,index=index,seconds=time.time()-begin,nodes=shift,targets=len(combined['child']),
        views=15,audit=dict(total),metadata_sha256=h.hexdigest(),features_sha256=sha(dest/'features.npz'),views_sha256=sha(dest/'VIEWS.json'),
        inputs=inputs,image_hashes=image_hashes,encoder_sha256=sha(ENCODERS[ds]),seal_sha256=sha(ROOT/'SEALED.json'),
        independent_target_checks=len(combined['child']),encoder_probes=probecount,max_encoder_probe_error=maxerr,device=gpu))
    print('PREPARED',ds,source,index,len(combined['child']),round(time.time()-begin,2),flush=True)

def worker(shard):
    import torch
    from encoder import Encoder
    torch.set_num_threads(1);torch.manual_seed(20261025);seal=check();resource=read(ROOT/'RESOURCE.json')
    gpu='cuda:0' if resource['gpu'] is not None else 'cpu'
    if gpu!='cpu':torch.cuda.set_per_process_memory_fraction(.10)
    encoders={ds:Encoder(ENCODERS[ds]).to(gpu) for ds in SPECS};cpu={ds:Encoder(ENCODERS[ds]) for ds in SPECS}
    for i,j in enumerate(seal['jobs']):
        if i%resource['workers']==shard:video_cache(j,gpu,encoders[j['ds']],cpu[j['ds']])

def assemble():
    seal=check();specs=read(ROOT/'SOURCE_SPEC.json')
    for ds,sources in specs.items():
        bindings={};counts={};metadata={};imagehash={};es=np.zeros(15);eq=np.zeros(15);en=0;ns=np.zeros(6);nq=np.zeros(6);nn=0
        for source,vs in sources.items():
            pools={(gap,k):[] for gap in GAPS for k in range(3)};fps=[];images=[];totals=collections.Counter()
            for i in range(len(vs)):
                d=ROOT/ds/'cache'/source/f'{i:03d}';r=read(d/'DONE.json')
                assert r['features_sha256']==sha(d/'features.npz') and r['views_sha256']==sha(d/'VIEWS.json')
                assert r['seal_sha256']==sha(ROOT/'SEALED.json') and r['encoder_sha256']==sha(ENCODERS[ds])
                with np.load(d/'features.npz') as z:
                    for gap,k in pools:
                        ii=np.flatnonzero((z['gap']==gap)&(z['kind']==k));pools[(gap,k)].extend((i,int(j)) for j in ii)
                    if source==ds+'_head':
                        valid=z['valid'].copy();valid[:,0]=False;x=z['edge_geometry'][valid].astype(float)
                        es+=x.sum(0);eq+=(x*x).sum(0);en+=len(x)
                        x=z['node_geometry'].astype(float);ns+=x.sum(0);nq+=(x*x).sum(0);nn+=len(x)
                totals.update(r['audit']);fps.append(r['metadata_sha256']);images.append(r['image_hashes']);bindings[str(d/'DONE.json')]=sha(d/'DONE.json')
            for (gap,k),rows in pools.items():
                p=ROOT/ds/'pools'/f'{source}_g{gap}_{k}.npy';p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:np.save(f,np.asarray(rows,np.int32).reshape(-1,2))
                bindings[str(p)]=sha(p)
            counts[source]=dict(videos=len(vs),audit=dict(totals),strata={f'g{g}_k{k}':len(p) for (g,k),p in pools.items()})
            metadata[source]=fps;imagehash[source]=images
        assert metadata[ds+'_G']==metadata[ds+'_Paint'],'Aligned/painting metadata differs'
        assert imagehash[ds+'_G']==imagehash[ds+'_Raw'],'Generated/Raw image identity differs'
        em=es/en;ed=np.maximum(np.sqrt(np.maximum(eq/en-em**2,0)),.1);nm=ns/nn;nd=np.maximum(np.sqrt(np.maximum(nq/nn-nm**2,0)),.1)
        for i in [3,4,5,9,10,11]:em[i]=0;ed[i]=1
        for i in [3,4,5]:nm[i]=0;nd[i]=1
        save(ROOT/ds/'scaling.npz',edge_mean=em.astype(np.float32),edge_sd=ed.astype(np.float32),node_mean=nm.astype(np.float32),node_sd=nd.astype(np.float32))
        bindings[str(ROOT/ds/'scaling.npz')]=sha(ROOT/ds/'scaling.npz')
        put(ROOT/ds/'PREPARED.json',dict(time=time.time(),counts=counts,bindings=bindings,exact_G_Paint_metadata=True,exact_G_Raw_images=True,seal_sha256=sha(ROOT/'SEALED.json')))
    put(ROOT/'DATA_READY.json',dict(time=time.time(),videos=1175,views=17625,prepared={ds:sha(ROOT/ds/'PREPARED.json') for ds in SPECS},status='PREPARED_WITH_ALL_TARGET_CHECKS_AND_SAMPLED_ENCODER_CHECKS'))

def driver():
    lock=(ROOT/'DRIVER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);seal()
    r=device();r.update(time=time.time(),workers=2 if r['gpu'] is not None else 6);put(ROOT/'RESOURCE.json',r)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if r['gpu'] is None else str(r['gpu']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    active=[]
    for shard in range(r['workers']):
        f=(ROOT/f'worker_{shard}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(HERE/'prepare.py'),'worker',str(shard)],stdout=f,stderr=subprocess.STDOUT,env=env);active.append((p,f))
    failures=[]
    for p,f in active:
        code=p.wait();f.close()
        if code:failures.append(dict(pid=p.pid,code=code))
    assert not failures,failures
    assemble()

if __name__=='__main__':
    try:
        if sys.argv[1]=='driver':driver()
        else:worker(int(sys.argv[2]))
    except BaseException:
        import traceback
        put(ROOT/f'FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
