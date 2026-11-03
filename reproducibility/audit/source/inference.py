"""Observed-only triplet scores with independent geometry/network reconstruction."""
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from triplets import descriptors

def independent_geometry(data,idx):
    rows=[]
    for p,a,b in idx:
        ap,aa,ab=map(float,data['area'][[p,a,b]])
        xy=np.asarray([[data['cy'][i],data['cx'][i]] for i in [p,a,b]],dtype=float)
        u=xy[1]-xy[0];v=xy[2]-xy[0];r=np.sqrt(ap/np.pi);la=float(np.sqrt(np.dot(u,u)));lb=float(np.sqrt(np.dot(v,v)))
        center=(aa*u+ab*v)/(aa+ab);ng=data['node_geometry'][[p,a,b]].astype(float)
        row=[np.log((aa+ab)/ap),abs(np.log(aa/ab)),float(np.sqrt(np.dot(center,center)))/r,
             float(np.sqrt(np.dot(u-v,u-v)))/r,min(la,lb)/r,max(la,lb)/r,float(np.dot(u,v))/max(la*lb,1e-8)]
        rows.append(row+ng[0].tolist()+((ng[1]+ng[2])/2).tolist()+abs(ng[1]-ng[2]).tolist())
    return np.clip(np.asarray(rows,np.float32),-12,12)

def independent_network(state,m,a,b,g):
    project=lambda x:F.gelu(F.linear(x,state['project.0.weight'],state['project.0.bias']))
    m,a,b=map(project,[m,a,b]);mean=(a+b)/2
    x=torch.cat([m,mean,abs(a-b),a*b,abs(m-mean),m*mean,g],dim=-1)
    x=F.gelu(F.linear(x,state['classifier.0.weight'],state['classifier.0.bias']))
    x=F.gelu(F.linear(x,state['classifier.3.weight'],state['classifier.3.bias']))
    return F.linear(x,state['classifier.5.weight'],state['classifier.5.bias']).squeeze(-1)

def score_graph(graph,model,arm,features,batch=1024):
    assert not model.training
    keys=list(zip(map(int,features['t']),map(int,features['seg_id'])));assert len(keys)==len(set(keys))
    lookup={key:i for i,key in enumerate(keys)}
    nodes={n:lookup[(int(v['time']),int(v['label']))] for n,v in graph.nodes(data=True)}
    pairs=[(p,a,b) for p in sorted(graph) for a,b in itertools.combinations(sorted(graph.successors(p)),2)]
    assert all(graph.nodes[a]['time']==graph.nodes[b]['time']==graph.nodes[p]['time']+1 for p,a,b in pairs)
    logits=[];max_geometry_error=0.;state=model.state_dict()
    observed={k:features[k] for k in ['area','cy','cx','node_geometry']}
    with torch.inference_mode():
        for first in range(0,len(pairs),batch):
            pp=pairs[first:first+batch];idx=np.asarray([[nodes[p],nodes[a],nodes[b]] for p,a,b in pp],np.int64)
            geometry=descriptors(observed,idx);independent=independent_geometry(observed,idx)
            np.testing.assert_allclose(geometry,independent,rtol=0,atol=1e-5);max_geometry_error=max(max_geometry_error,float(np.max(abs(geometry-independent))))
            emb=features['embedding'][idx].astype(np.float32)
            if arm=='Geo':emb[:]=0
            m,a,b=[torch.from_numpy(emb[:,i]) for i in range(3)];g=torch.from_numpy(geometry)
            out=model(m,a,b,g);swapped=model(m,b,a,g);manual=independent_network(state,m,a,b,g)
            torch.testing.assert_close(out,swapped,rtol=0,atol=0);torch.testing.assert_close(out,manual,rtol=0,atol=1e-6)
            logits.append(out.numpy())
    values=np.concatenate(logits) if logits else np.empty(0,np.float32);assert np.isfinite(values).all()
    return dict(mother=np.asarray([p for p,a,b in pairs],np.int64),daughter_a=np.asarray([a for p,a,b in pairs],np.int64),daughter_b=np.asarray([b for p,a,b in pairs],np.int64),
        logit=values,residual=np.clip(values,-4,4),maximum_geometry_error=np.asarray(max_geometry_error))
