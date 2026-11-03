"""Observable parent candidates; identity supervision is separate from features."""
import collections
import numpy as np
from scipy.spatial import cKDTree
KEYS={'t','cy','cx','area'}

def node_geometry(obs):
 assert set(obs)==KEYS
 area=np.asarray(obs['area'],float);assert np.all(area>0)
 xy=np.column_stack([obs['cy'],obs['cx']]);radius=np.sqrt(area/np.pi);out=np.zeros((len(area),3),np.float32);out[:,0]=np.log(area)
 for t in sorted(set(obs['t'])):
  ii=np.flatnonzero(obs['t']==t);tree=cKDTree(xy[ii]);r=radius[ii]
  if len(ii)>1:out[ii,1]=tree.query(xy[ii],k=2)[0][:,1]/r
  out[ii,2]=[len(v)-1 for v in tree.query_ball_point(xy[ii],4*r)]
 return out

def candidate_parents(obs,k=10,distance=256.):
 assert set(obs)==KEYS
 xy=np.column_stack([obs['cy'],obs['cx']]);result=[]
 for t in sorted(set(obs['t'])):
  parents=np.flatnonzero(obs['t']==t);kids=np.flatnonzero(obs['t']==t+1)
  if not len(parents) or not len(kids):continue
  n=min(k,len(parents));dd,ix=cKDTree(xy[parents]).query(xy[kids],k=n)
  for c,ds,ids in zip(kids,np.asarray(dd).reshape(-1,n),np.asarray(ix).reshape(-1,n)):
   selected=sorted([(float(d),int(parents[j])) for d,j in zip(ds,ids) if d<=distance])
   result.append((int(c),tuple(p for d,p in selected)))
 return result

def supervised(obs,track_ids,tracks):
 choices=candidate_parents(obs);ids=np.asarray(track_ids);lut=collections.defaultdict(list);offspring=collections.Counter(a for b,e,a in tracks.values() if a)
 for i,(t,k) in enumerate(zip(obs['t'],ids)):
  if k>0:lut[(int(t),int(k))].append(i)
 rows=[];audit=collections.Counter()
 for c,parents in choices:
  k=int(ids[c]);t=int(obs['t'][c]);audit['children']+=1
  if k<=0 or k not in tracks:audit['unknown_child']+=1;continue
  b,e,a=tracks[k];assert b<=t<=e
  if b<t:expected=k;kind=0
  elif a==0:expected=0;kind=2
  elif a not in tracks or tracks[a][1]!=t-1:audit['invalid_parent_timing']+=1;continue
  else:expected=a;kind=1 if offspring[a]>=2 else 0
  if expected==0:target=0
  else:
   support=lut.get((t-1,expected),[])
   if len(support)!=1:audit['missing_or_ambiguous_true_parent']+=1;continue
   if support[0] not in parents:audit['true_parent_outside_candidates']+=1;continue
   target=parents.index(support[0])+1
  padded=[-1]+list(parents)+[-1]*(10-len(parents));valid=[True]+[True]*len(parents)+[False]*(10-len(parents))
  rows.append((c,padded,valid,target,kind));audit[['continuation','daughter','entry'][kind]]+=1
 if not rows:return dict(child=np.empty(0,np.int32),parents=np.empty((0,11),np.int32),valid=np.empty((0,11),bool),target=np.empty(0,np.int64),kind=np.empty(0,np.int8)),dict(audit)
 c,pp,vv,y,kind=zip(*rows)
 return dict(child=np.asarray(c,np.int32),parents=np.asarray(pp,np.int32),valid=np.asarray(vv,bool),target=np.asarray(y,np.int64),kind=np.asarray(kind,np.int8)),dict(audit)

def pair_geometry(obs,nodes,parent,child):
 assert set(obs)==KEYS
 p=np.asarray(parent);c=np.asarray(child);pa=obs['area'][p];ca=obs['area'][c]
 d=np.hypot(obs['cy'][p]-obs['cy'][c],obs['cx'][p]-obs['cx'][c])
 return np.column_stack([nodes[p],nodes[c],np.log(ca/pa),d/np.sqrt(pa/np.pi),d/np.sqrt(ca/np.pi)]).astype(np.float32)

def draw(pool,rng,n):
 groups=[np.flatnonzero(pool['kind']==i) for i in range(3)];assert any(len(g) for g in groups)
 counts=[n-2*(n//4),n//4,n//4];out=[]
 for i,count in enumerate(counts):
  group=groups[i] if len(groups[i]) else next(g for g in groups if len(g))
  out.extend(group[rng.integers(0,len(group),count)].tolist())
 return np.array(out,np.int64)
