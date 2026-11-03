"""Equal-budget50 triplet heads from sealed source-only caches."""
from pathlib import Path
import collections,concurrent.futures,fcntl,functools,hashlib,json,math,os,subprocess,sys,time
import numpy as np
ROOT=Path('STUDY_ROOT/eval/mitosync/division_triplet_20261025/v1')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from prepare import MR,sha,read,put
from triplets import descriptors
ARMS=['Geo','R','RG','PM','Raw'];STEPS=6000

def sources(ds,arm):
    return [(ds+'_head',64)] if arm in ['Geo','R'] else [(ds+'_head',32),(ds+'_'+{'RG':'G','PM':'Paint','Raw':'Raw'}[arm],32)]

def digest(h,rows):
    for source,i,j in rows:
        source=source.replace('_Paint','_matched').replace('_G','_matched')
        h.update(f'{source}:{i}:{j}\n'.encode())

def draw(pools,ds,arm,rng):
    rows=[];counts=collections.Counter()
    for source,n in sources(ds,arm):
        for gap in [1,2,4,8]:
            for y in [0,1]:
                pool=pools[source,gap,y];ix=rng.integers(0,len(pool),n//8)
                rows.extend((source,int(i),int(j)) for i,j in pool[ix]);counts[f'{source}/g{gap}/y{y}']+=len(ix)
    assert len(rows)==64
    return rows,counts

class Data:
    def __init__(self,ds):
        self.ds=ds;self.ready=read(ROOT/'DATA_READY.json');self.pools={}
        for source in [ds+'_'+s for s in ['head','G','Paint','Raw']]:
            for g in [1,2,4,8]:
                for y in [0,1]:
                    p=ROOT/'pools'/f'{source}_g{g}_y{y}.npy';assert sha(p)==self.ready['bindings'][str(p)]
                    self.pools[source,g,y]=np.load(p)
    @functools.lru_cache(maxsize=512)
    def video(self,source,i):
        d=ROOT/'cache'/self.ds/source/f'{i:03d}';r=read(d/'DONE.json')
        assert sha(d/'DONE.json')==self.ready['receipts'][str(d/'DONE.json')]
        assert sha(d/'triplets.npy')==r['triplets_sha256']
        old=MR/self.ds/'cache'/source/f'{i:03d}';assert sha(old/'features.npz')==r['source_features_sha256']
        with np.load(old/'features.npz') as z:data={k:z[k] for k in ['embedding','area','cy','cx','node_geometry']}
        return data,np.load(d/'triplets.npy')
    def batch(self,arm,rng):
        import torch
        rows,counts=draw(self.pools,self.ds,arm,rng);groups=collections.defaultdict(list)
        for k,(source,i,j) in enumerate(rows):groups[source,i].append((k,j))
        embeddings=np.zeros((64,3,256),np.float32);geometry=np.empty((64,25),np.float32);labels=np.empty(64,np.float32)
        for (source,i),items in groups.items():
            data,triples=self.video(source,i);kk,jj=np.asarray(items).T;selected=triples[jj]
            if arm!='Geo':embeddings[kk]=data['embedding'][selected[:,:3]]
            geometry[kk]=descriptors(data,selected[:,:3]);labels[kk]=selected[:,3]
        return [torch.from_numpy(v) for v in [embeddings[:,0],embeddings[:,1],embeddings[:,2],geometry,labels]],rows,counts

def seal():
    ready=read(ROOT/'DATA_READY.json');assert ready['videos']==1175 and ready['matched_G_Paint_exact']
    inputs={str(ROOT/'DATA_READY.json'):sha(ROOT/'DATA_READY.json'),str(ROOT/'SEALED.json'):sha(ROOT/'SEALED.json'),**ready['receipts'],**ready['bindings']}
    plans={}
    for ds in ['psc','fluo']:
        data=Data(ds)
        for g in [1,2,4,8]:
            for y in [0,1]:np.testing.assert_array_equal(data.pools[ds+'_G',g,y],data.pools[ds+'_Paint',g,y])
        for arm in ARMS:
            for seed in range(1,6):
                rng=np.random.default_rng(seed);h=hashlib.sha256();counts=collections.Counter()
                # Independently reconstruct pool indices; no Data.batch assembly.
                for _ in range(STEPS):
                    rows=[]
                    for source,n in sources(ds,arm):
                        for gap in (1,2,4,8):
                            for label in (0,1):
                                pool=data.pools[source,gap,label];ii=rng.integers(0,len(pool),n//8)
                                rows.extend((source,int(i),int(j)) for i,j in pool[ii]);counts[f'{source}/g{gap}/y{label}']+=len(ii)
                    digest(h,rows)
                plans[f'{ds}_{arm}_s{seed}']=dict(sample_digest=h.hexdigest(),sample_counts=dict(counts))
        for seed in range(1,6):assert plans[f'{ds}_RG_s{seed}']['sample_digest']==plans[f'{ds}_PM_s{seed}']['sample_digest']
    for p,h in inputs.items():assert sha(p)==h,p
    bound={str(p):sha(p) for folder in [ROOT/'source',HERE] for p in folder.iterdir() if p.suffix in ['.py','.md']}
    put(ROOT/'TRAINING_SEALED.json',dict(time=time.time(),inputs=inputs,sources=bound,plans=plans,models=50,updates=6000,batch=64,
        optimizer='AdamW',lr=.001,weight_decay=.01,loss='balanced BCEWithLogits',no_checkpoint_selection=True))

def train(ds,arm,seed):
    import torch
    from model import TripletModel
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    sealed=read(ROOT/'TRAINING_SEALED.json')
    for p,h in sealed['sources'].items():assert sha(p)==h,p
    d=ROOT/'models'/f'{ds}_{arm}_s{seed}';d.mkdir(parents=True,exist_ok=False)
    put(d/'STARTED.json',dict(time=time.time(),pid=os.getpid(),ds=ds,arm=arm,seed=seed,training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json')))
    data=Data(ds);net=TripletModel();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:.5*(1+math.cos(math.pi*t/STEPS)))
    h=hashlib.sha256();counts=collections.Counter();history=[];begin=time.time()
    for step in range(STEPS):
        (m,a,b,g,y),rows,cc=data.batch(arm,rng);digest(h,rows);counts.update(cc)
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(net(m,a,b,g),y)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.);assert torch.isfinite(norm)
        opt.step();scheduler.step()
        if step%100==0 or step==STEPS-1:
            v=dict(step=step+1,loss=float(loss.detach()),seconds=time.time()-begin);history.append(v)
            tmp=d/'PROGRESS.tmp';tmp.write_text(json.dumps(v));tmp.replace(d/'PROGRESS.json')
    expected=sealed['plans'][f'{ds}_{arm}_s{seed}']
    assert h.hexdigest()==expected['sample_digest'] and dict(counts)==expected['sample_counts']
    assert all(int(v['step'])==STEPS for v in opt.state.values()) and all(torch.isfinite(p).all() for p in net.parameters())
    torch.save(dict(state_dict=net.state_dict(),ds=ds,arm=arm,seed=seed,updates=STEPS,geometry_dim=25),d/'model.pt')
    put(d/'DONE.json',dict(time=time.time(),ds=ds,arm=arm,seed=seed,updates=STEPS,seconds=time.time()-begin,history=history,
        sample_digest=h.hexdigest(),sample_counts=dict(counts),model_sha256=sha(d/'model.pt'),training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json')))

def task(job):
    ds,arm,seed=job
    with (ROOT/f'train_{ds}_{arm}_s{seed}.log').open('x') as f:
        subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'train',ds,arm,str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)

def driver():
    lock=(ROOT/'TRAINING.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'DATA_READY.json').exists():
        assert not list(ROOT.glob('FAILURE*.json')),'Preparation failure; preserve and stop'
        pid=read(ROOT/'START_REQUEST.json')['pid'];p=Path('/proc')/str(pid)
        assert p.exists() and str(ROOT/'source/prepare.py').encode() in (p/'cmdline').read_bytes()
        assert (p/'stat').read_text().rsplit(')',1)[1].strip().split()[0]!='Z'
        time.sleep(20)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>32*1024**3
    seal();jobs=iter((ds,a,s) for s in range(1,6) for ds in ['psc','fluo'] for a in ARMS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        active={pool.submit(task,next(jobs)) for _ in range(6)}
        while active:
            done,active=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:f.result()
            for _ in done:
                j=next(jobs,None)
                if j is not None:active.add(pool.submit(task,j))
    sealed=read(ROOT/'TRAINING_SEALED.json')
    for p,h in {**sealed['inputs'],**sealed['sources']}.items():assert sha(p)==h,p
    rr=list((ROOT/'models').glob('*/DONE.json'));assert len(rr)==50;bindings={}
    for p in rr:
        v=read(p);assert v['model_sha256']==sha(p.parent/'model.pt');bindings[str(p)]=sha(p);bindings[str(p.parent/'model.pt')]=v['model_sha256']
    put(ROOT/'ALL_MODELS_TRAINED.json',dict(time=time.time(),models=50,bindings=bindings,training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json')))

if __name__=='__main__':
    try:
        if sys.argv[1]=='driver':driver()
        else:train(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    except BaseException:
        import traceback
        put(ROOT/f'TRAINING_FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
