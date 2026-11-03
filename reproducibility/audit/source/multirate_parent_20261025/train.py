"""Equal-budget fifty-head training after complete multirate preparation."""
import collections,concurrent.futures,fcntl,functools,hashlib,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
E=Path('STUDY_ROOT/eval/mitosync')
ROOT=E/'multirate_parent_20261025/v1';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from common import sha,read,put,GAPS
ARMS=['Geo','R','RG','PM','Raw'];STEPS=6000

def sources(ds,arm):
    assert ds in ['psc','fluo'] and arm in ARMS
    return [(ds+'_head',64)] if arm in ['Geo','R'] else [(ds+'_head',32),(ds+'_'+{'RG':'G','PM':'Paint','Raw':'Raw'}[arm],32)]

def digest_rows(h,rows):
    for s,v,r in rows:
        s=s.replace('_Paint','_matched').replace('_G','_matched')
        h.update(f'{s}:{v}:{r}\n'.encode())

class Data:
    def __init__(self,ds):
        self.ds=ds;self.root=ROOT/ds;self.meta=read(self.root/'PREPARED.json')
        for p in [self.root/'scaling.npz']+list((self.root/'pools').glob('*.npy')):assert sha(p)==self.meta['bindings'][str(p)]
        self.scale=dict(np.load(self.root/'scaling.npz'))
        self.pools={(s,g,k):np.load(self.root/'pools'/f'{s}_g{g}_{k}.npy') for s in self.meta['counts'] for g in GAPS for k in range(3)}
        for s in self.meta['counts']:
            for g in GAPS:assert any(len(self.pools[s,g,k]) for k in range(3)),(s,g)
    @functools.lru_cache(maxsize=512)
    def video(self,s,i):
        d=self.root/'cache'/s/f'{i:03d}';r=d/'DONE.json'
        assert sha(r)==self.meta['bindings'][str(r)] and sha(d/'features.npz')==read(r)['features_sha256']
        with np.load(d/'features.npz') as z:return {k:z[k].copy() for k in ['embedding','node_geometry','edge_geometry','child','parents','valid','target','kind','gap']}
    def batch(self,arm,rng):
        import torch
        arrays=[[] for _ in range(6)];samples=[];counts=collections.Counter()
        for source,n in sources(self.ds,arm):
            per=n//4
            for gap in GAPS:
                for k,count in enumerate([per//2,per//4,per//4]):
                    pool=self.pools[source,gap,k]
                    if not len(pool):pool=next(self.pools[source,gap,j] for j in range(3) if len(self.pools[source,gap,j]))
                    for vi,row in pool[rng.integers(0,len(pool),count)]:
                        d=self.video(source,int(vi));assert int(d['gap'][row])==gap
                        child=d['child'][row];pa=d['parents'][row];p=d['embedding'][np.maximum(pa,0)].astype(np.float32);p[pa<0]=0
                        c=d['embedding'][child].astype(np.float32)
                        if arm=='Geo':p[:]=0;c[:]=0
                        values=[p,c,np.clip((d['edge_geometry'][row].astype(np.float32)-self.scale['edge_mean'])/self.scale['edge_sd'],-8,8),
                            np.clip((d['node_geometry'][child].astype(np.float32)-self.scale['node_mean'])/self.scale['node_sd'],-8,8),d['valid'][row],d['target'][row]]
                        for a,v in zip(arrays,values):a.append(v)
                        samples.append((source,int(vi),int(row)));counts[f'{source}/g{gap}/k{int(d["kind"][row])}']+=1
        return [torch.from_numpy(np.asarray(a)) for a in arrays],samples,counts

def prepare():
    ready=read(ROOT/'DATA_READY.json');assert ready['videos']==1175 and ready['views']==17625
    inputs={str(ROOT/'DATA_READY.json'):sha(ROOT/'DATA_READY.json')};plans={}
    for ds in ['psc','fluo']:
        p=ROOT/ds/'PREPARED.json';assert sha(p)==ready['prepared'][ds];meta=read(p)
        assert meta['exact_G_Paint_metadata'] and meta['exact_G_Raw_images'];inputs[str(p)]=sha(p);inputs.update(meta['bindings'])
        pools={(s,g,k):np.load(ROOT/ds/'pools'/f'{s}_g{g}_{k}.npy') for s in meta['counts'] for g in GAPS for k in range(3)}
        for g in GAPS:
            for k in range(3):np.testing.assert_array_equal(pools[ds+'_G',g,k],pools[ds+'_Paint',g,k])
        for arm in ARMS:
            for seed in range(1,6):
                rng=np.random.default_rng(seed);h=hashlib.sha256();counts=collections.Counter()
                # Direct pool draws, independent of Data/video/tensor assembly.
                for step in range(STEPS):
                    rows=[]
                    for source,n in sources(ds,arm):
                        for gap in (1,2,4,8):
                            for kind,number in enumerate((n//8,n//16,n//16)):
                                actual=kind
                                if not len(pools[source,gap,actual]):actual=next(k for k in range(3) if len(pools[source,gap,k]))
                                pool=pools[source,gap,actual];ix=rng.integers(0,len(pool),number)
                                rows.extend((source,int(v),int(r)) for v,r in pool[ix]);counts[f'{source}/g{gap}/k{actual}']+=number
                    digest_rows(h,rows)
                plans[f'{ds}_{arm}_s{seed}']=dict(sample_digest=h.hexdigest(),sample_counts=dict(counts))
        for seed in range(1,6):assert plans[f'{ds}_RG_s{seed}']['sample_digest']==plans[f'{ds}_PM_s{seed}']['sample_digest']
    for p,h in inputs.items():assert sha(p)==h,p
    sources_bound={str(p):sha(p) for folder in [HERE,ROOT/'source'] for p in folder.glob('*.py')}
    put(ROOT/'TRAINING_SEALED.json',dict(time=time.time(),inputs=inputs,sources=sources_bound,plans=plans,updates=6000,batch=64,models=50,
        protocol_sha256=sha(ROOT/'source/PROTOCOL.md')))

def train(ds,arm,seed):
    import torch
    from model import ParentModel
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    seal=read(ROOT/'TRAINING_SEALED.json')
    for p,h in seal['sources'].items():assert sha(p)==h,p
    assert sha(ROOT/ds/'PREPARED.json')==seal['inputs'][str(ROOT/ds/'PREPARED.json')]
    dest=ROOT/'models'/f'{ds}_{arm}_s{seed}';dest.mkdir(parents=True,exist_ok=False)
    put(dest/'STARTED.json',dict(time=time.time(),pid=os.getpid(),ds=ds,arm=arm,seed=seed,seal_sha256=sha(ROOT/'TRAINING_SEALED.json')))
    data=Data(ds);net=ParentModel();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.5*(1+math.cos(math.pi*s/STEPS)))
    h=hashlib.sha256();counts=collections.Counter();losses=[];begin=time.time()
    for step in range(STEPS):
        tensors,samples,cc=data.batch(arm,rng);digest_rows(h,samples);counts.update(cc);p,c,e,n,v,y=tensors
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(p,c,e,n,v),y.long());assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.);assert torch.isfinite(norm)
        opt.step();sched.step()
        if step%100==0 or step==STEPS-1:
            progress=dict(step=step+1,loss=float(loss.detach()),seconds=time.time()-begin);losses.append(progress)
            q=dest/'PROGRESS.tmp';q.write_text(__import__('json').dumps(progress));q.replace(dest/'PROGRESS.json')
    expected=seal['plans'][f'{ds}_{arm}_s{seed}'];assert h.hexdigest()==expected['sample_digest'] and dict(counts)==expected['sample_counts']
    assert all(int(s['step'])==STEPS for s in opt.state.values()) and all(torch.isfinite(p).all() for p in net.parameters())
    torch.save(dict(state_dict=net.state_dict(),ds=ds,arm=arm,seed=seed,updates=STEPS,node_dim=6,edge_dim=15),dest/'model.pt')
    put(dest/'DONE.json',dict(time=time.time(),ds=ds,arm=arm,seed=seed,updates=STEPS,batch=64,seconds=time.time()-begin,
        sample_digest=h.hexdigest(),sample_counts=dict(counts),model_sha256=sha(dest/'model.pt'),seal_sha256=sha(ROOT/'TRAINING_SEALED.json'),loss_samples=losses))

def task(ds,arm,seed):
    with (ROOT/f'train_{ds}_{arm}_s{seed}.log').open('x') as f:
        subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'train',ds,arm,str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)

def driver():
    lock=(ROOT/'TRAINING.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'DATA_READY.json').exists():
        assert not list(ROOT.glob('FAILURE*')),'Preparation failure preserved; no auto restart'
        pid=read(ROOT/'START_REQUEST.json')['pid'];p=Path('/proc')/str(pid)
        assert p.exists() and str(ROOT/'source/prepare.py').encode() in (p/'cmdline').read_bytes()
        assert (p/'stat').read_text().rsplit(')',1)[1].strip().split()[0]!='Z'
        time.sleep(20)
    prepare()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(task,ds,arm,s) for s in range(1,6) for ds in ['psc','fluo'] for arm in ARMS]
        for f in concurrent.futures.as_completed(fs):f.result()
    seal=read(ROOT/'TRAINING_SEALED.json')
    for p,h in {**seal['inputs'],**seal['sources']}.items():assert sha(p)==h,p
    receipts=list((ROOT/'models').glob('*/DONE.json'));assert len(receipts)==50;bindings={}
    for p in receipts:
        r=read(p);assert sha(p.parent/'model.pt')==r['model_sha256'];bindings[str(p)]=sha(p);bindings[str(p.parent/'model.pt')]=r['model_sha256']
    put(ROOT/'ALL_MODELS_TRAINED.json',dict(time=time.time(),models=50,seal_sha256=sha(ROOT/'TRAINING_SEALED.json'),bindings=bindings))

if __name__=='__main__':
    try:
        if sys.argv[1]=='driver':driver()
        else:train(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    except BaseException:
        import traceback
        put(ROOT/f'TRAINING_FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
