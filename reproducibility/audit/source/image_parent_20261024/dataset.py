"""Matched source sampling of cached parent-choice examples."""
import collections,functools,hashlib
import numpy as np
import torch
from data_common import ROOT,read,sha
SYN={'RG':'psc_G','PM':'psc_Paint','Raw':'psc_Raw'}

class Data:
 def __init__(self,root=ROOT):
  self.root=root;self.meta=read(root/'PREPARED.json')
  for p in [root/'scaling.npz']+list((root/'pools').glob('*.npy')):assert sha(p)==self.meta['bindings'][str(p)],p
  self.scaling=dict(np.load(root/'scaling.npz'))
  self.pools={s:[np.load(root/'pools'/f'{s}_{k}.npy') for k in range(3)] for s in self.meta['counts']}
 @functools.lru_cache(maxsize=256)
 def video(self,source,index):
  d=self.root/'cache'/source/f'{index:03d}';receipt=d/'DONE.json'
  assert sha(receipt)==self.meta['bindings'][str(receipt)],receipt
  assert sha(d/'features.npz')==read(receipt)['features_sha256'],d
  with np.load(d/'features.npz') as z:return {k:z[k].copy() for k in ['embedding','node_geometry','edge_geometry','child','parents','valid','target','kind']}
 def indices(self,source,n,rng):
  pools=self.pools[source];counts=[n-2*(n//4),n//4,n//4];rows=[]
  for k,count in enumerate(counts):
   p=pools[k] if len(pools[k]) else next(p for p in pools if len(p))
   rows.append(p[rng.integers(0,len(p),count)])
  return np.concatenate(rows)
 def batch(self,arm,rng):
  sources=[('psc_head',64)] if arm in ['R','Geo'] else [('psc_head',32),(SYN[arm],32)]
  parent=[];child=[];edge=[];node=[];valid=[];target=[];samples=[];counts=collections.Counter()
  for source,n in sources:
   indexes=self.indices(source,n,rng)
   for video,row in indexes:
    d=self.video(source,int(video));c=d['child'][row];pp=d['parents'][row];v=d['valid'][row];e=d['embedding'][np.maximum(pp,0)].astype(np.float32);e[pp<0]=0
    z=d['embedding'][c].astype(np.float32)
    if arm=='Geo':e[:]=0;z[:]=0
    parent.append(e);child.append(z);edge.append(np.clip((d['edge_geometry'][row].astype(np.float32)-self.scaling['edge_mean'])/self.scaling['edge_sd'],-8,8))
    node.append(np.clip((d['node_geometry'][c].astype(np.float32)-self.scaling['node_mean'])/self.scaling['node_sd'],-8,8))
    valid.append(v);target.append(d['target'][row]);samples.append((source,int(video),int(row)));counts[f'{source}/kind{int(d["kind"][row])}']+=1
  tensors=[torch.from_numpy(np.asarray(v)) for v in [parent,child,edge,node,valid,target]]
  return tensors,samples,counts

def sample_digest_update(h,samples):
 # RG and PM equivalent source names are canonicalized for direct paired checking.
 for source,index,row in samples:
  source='matched_synthetic' if source in ['psc_G','psc_Paint'] else source
  h.update(f'{source}:{index}:{row}\n'.encode())
