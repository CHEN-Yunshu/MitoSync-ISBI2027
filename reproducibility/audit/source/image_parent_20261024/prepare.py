"""Full-source ROI features and identity-conditioned parent targets, CPU only."""
import argparse,collections,fcntl,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import tifffile
from data_common import *
from encoder import Encoder,crop_masks
from targets import node_geometry,pair_geometry,supervised

def video_cache(source,index,encoder):
 spec=read(SPEC);v=spec['sources'][source]['videos'][index];dest=ROOT/'cache'/source/f'{index:03d}'
 assert not dest.exists(),('Explicit completed-boundary recovery required',dest);dest.mkdir(parents=True)
 started=time.time();put(dest/'STARTED.json',dict(time=started,pid=os.getpid(),source=source,index=index,name=v['name']))
 inputs={v['shard']:sha(v['shard'])};trackfile=Path(v['tra_dir'])/'man_track.txt';inputs[str(trackfile)]=sha(trackfile)
 with np.load(v['shard']) as z:
  obs={k:z[k].copy() for k in OBS};ids=z['trk'].copy();seg_id=z['seg_id'].copy();crops=z['crops'];ys=obs['cy'];xs=obs['cx']
 masks=np.zeros((len(ids),48,48),bool);mask_sha=hashlib.sha256()
 for t in sorted(set(obs['t'])):
  path=Path(v['seg_dir'])/f'man_seg{int(t):03d}.tif';inputs[str(path)]=sha(path);seg=tifffile.imread(path)
  ii=np.flatnonzero(obs['t']==t);masks[ii]=crop_masks(seg,seg_id[ii],ys[ii],xs[ii])
 mask_sha.update(masks.tobytes());emb=[]
 with torch.inference_mode():
  for i in range(0,len(crops),128):emb.append(encoder(torch.from_numpy(crops[i:i+128]),torch.from_numpy(masks[i:i+128])).numpy().astype(np.float16))
 emb=np.concatenate(emb);assert emb.shape==(len(ids),256) and np.isfinite(emb).all()
 nodes=node_geometry(obs);tracks={int(k):(int(b),int(e),int(p)) for k,b,e,p in np.loadtxt(trackfile,dtype=int,ndmin=2)}
 rows,audit=supervised(obs,ids,tracks);n=len(rows['child']);eg=np.zeros((n,11,9),np.float32)
 valid=rows['valid'].copy();valid[:,0]=False;rr,cc=np.nonzero(valid)
 if len(rr):eg[rr,cc]=pair_geometry(obs,nodes,rows['parents'][rr,cc],rows['child'][rr])
 arrays=dict(**obs,embedding=emb,node_geometry=nodes.astype(np.float16),edge_geometry=eg.astype(np.float16),**rows)
 save(dest/'features.npz',**arrays)
 h=hashlib.sha256()
 for k in OBS+['node_geometry','edge_geometry','child','parents','valid','target','kind']:h.update(np.ascontiguousarray(arrays[k]).tobytes())
 h.update(ids.tobytes());h.update(seg_id.tobytes());h.update(trackfile.read_bytes());h.update(masks.tobytes())
 images={p.name:sha(p) for p in sorted(Path(v['img_dir']).glob('t*.tif'))};assert len(images)==v['T']
 put(dest/'DONE.json',dict(source=source,index=index,name=v['name'],started_unix=started,finished_unix=time.time(),seconds=time.time()-started,nodes=len(ids),targets=n,audit=audit,inputs=inputs,image_hashes=images,
  metadata_sha256=h.hexdigest(),mask_sha256=mask_sha.hexdigest(),features_sha256=sha(dest/'features.npz'),encoder_sha256=sha(ENCODER),preparation_sha256=sha(ROOT/'PREPARATION_SEALED.json')))
 print('PREPARED',source,index,'nodes',len(ids),'targets',n,'seconds',round(time.time()-started,2),flush=True)

def worker(shard):
 torch.set_num_threads(1);torch.manual_seed(20261024);seal=check();encoder=Encoder(ENCODER)
 for i,job in enumerate(seal['jobs']):
  if i%3==shard:video_cache(job['source'],job['index'],encoder)

def assemble():
 seal=check();spec=read(SPEC);counts={};bindings={};fingerprints={};images={};head_sum=np.zeros(9);head_sq=head_sum.copy();head_n=0;node_sum=np.zeros(3);node_sq=node_sum.copy();node_n=0
 for source in SOURCES:
  totals=collections.Counter();fps=[];ims=[];listing=[];pools=[[],[],[]]
  for index,v in enumerate(spec['sources'][source]['videos']):
   d=ROOT/'cache'/source/f'{index:03d}';r=read(d/'DONE.json');assert r['features_sha256']==sha(d/'features.npz') and r['encoder_sha256']==sha(ENCODER)
   assert r['preparation_sha256']==sha(ROOT/'PREPARATION_SEALED.json')
   for p,h in r['inputs'].items():assert sha(p)==h,p
   with np.load(d/'features.npz') as z:
    for k in range(3):
     rows=np.flatnonzero(z['kind']==k);pools[k].extend((index,int(i)) for i in rows)
    if source=='psc_head':
     valid=z['valid'].copy();valid[:,0]=False;x=z['edge_geometry'][valid].astype(np.float64)
     head_sum+=x.sum(0);head_sq+=(x*x).sum(0);head_n+=len(x)
     x=z['node_geometry'].astype(np.float64);node_sum+=x.sum(0);node_sq+=(x*x).sum(0);node_n+=len(x)
   totals.update(r['audit']);fps.append(r['metadata_sha256']);ims.append(r['image_hashes']);listing.append(str(d/'features.npz'));bindings[str(d/'DONE.json')]=sha(d/'DONE.json')
  for k,rows in enumerate(pools):
   p=ROOT/'pools'/f'{source}_{k}.npy';p.parent.mkdir(exist_ok=True);assert not p.exists();np.save(p,np.asarray(rows,np.int32).reshape(-1,2));bindings[str(p)]=sha(p)
  counts[source]=dict(videos=len(listing),counts=dict(totals),stratum_sizes=[len(p) for p in pools],files=listing);fingerprints[source]=fps;images[source]=ims
 assert fingerprints['psc_G']==fingerprints['psc_Paint'],'RG/PM metadata, masks, geometry or targets differ'
 assert images['psc_G']==images['psc_Raw'],'Raw and RG underlying rendered images differ'
 mean=head_sum/head_n;sd=np.maximum(np.sqrt(np.maximum(head_sq/head_n-mean**2,0)),.1)
 nm=node_sum/node_n;ns=np.maximum(np.sqrt(np.maximum(node_sq/node_n-nm**2,0)),.1)
 save(ROOT/'scaling.npz',edge_mean=mean.astype(np.float32),edge_sd=sd.astype(np.float32),node_mean=nm.astype(np.float32),node_sd=ns.astype(np.float32));bindings[str(ROOT/'scaling.npz')]=sha(ROOT/'scaling.npz')
 put(ROOT/'DATA_COUNTS.json',counts);put(ROOT/'PREPARED.json',dict(status='PREPARED_PENDING_INDEPENDENT_AUDIT',time=time.time(),preparation_sha256=sha(ROOT/'PREPARATION_SEALED.json'),encoder_sha256=sha(ENCODER),counts=counts,bindings=bindings,RG_PM_exact_metadata_masks_targets=True,RG_Raw_same_underlying_images=True))
 print('ALL_DATA_PREPARED',flush=True)

def driver():
 lock=(ROOT/'PREPARATION.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 put(ROOT/'PREPARATION_LAUNCH.json',dict(time=time.time(),pid=os.getpid(),workers=3,threads_per_worker=1,cpu_only=True))
 active=[]
 for shard in range(3):
  log=(ROOT/f'prepare_{shard}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(HERE/'prepare.py'),'worker','--shard',str(shard)],stdout=log,stderr=subprocess.STDOUT);active.append((p,log,shard))
 failures=[]
 while active:
  time.sleep(5)
  for p,log,shard in list(active):
   if p.poll() is not None:
    log.close();active.remove((p,log,shard))
    if p.returncode:failures.append((shard,p.returncode))
 assert not failures,failures
 assemble()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['driver','worker','assemble']);ap.add_argument('--shard',type=int);a=ap.parse_args()
 try:
  if a.mode=='driver':driver()
  elif a.mode=='worker':worker(a.shard)
  else:assemble()
 except BaseException:
  import traceback
  put(ROOT/f'PREPARATION_FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
