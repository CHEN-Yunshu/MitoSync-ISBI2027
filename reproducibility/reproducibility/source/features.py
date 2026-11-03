"""Observed-only multirate context, including explicitly marked endpoints."""
import collections
import numpy as np
from encoder import crop_masks
from targets import node_geometry,supervised
from lineage_projection import project_tracks,norm_u8

def context(t,total):
    assert total>=2 and 0<=t<total
    return max(t-1,0),t,min(t+1,total-1)

def identities(seg,tra):
    """Existing deterministic max-overlap, one-to-one ownership convention."""
    assert seg.shape==tra.shape
    ids=np.unique(seg);ids=ids[ids>0];m=(seg>0)&(tra>0);best={}
    if m.any():
        keys,counts=np.unique(seg[m].astype(np.int64)*(1<<32)+tra[m].astype(np.int64),return_counts=True)
        for key,n in zip(keys,counts):
            s,q=int(key>>32),int(key&0xffffffff)
            if s not in best or n>best[s][1] or (n==best[s][1] and q<best[s][0]):best[s]=(q,int(n))
    owner={}
    for s in sorted(best):
        q,n=best[s]
        if q not in owner or n>best[owner[q]][1]:owner[q]=s
    return {int(s):best[s][0] if s in best and owner[best[s][0]]==s else 0 for s in ids}

def edge_geometry(obs,nodes,p,c):
    p=np.asarray(p);c=np.asarray(c);pa=obs['area'][p];ca=obs['area'][c]
    d=np.hypot(obs['cy'][p]-obs['cy'][c],obs['cx'][p]-obs['cx'][c])
    return np.column_stack([nodes[p],nodes[c],np.log(ca/pa),d/np.sqrt(pa/np.pi),d/np.sqrt(ca/np.pi)]).astype(np.float32)

def view_features(images,segs,encoder,gap,device='cpu'):
    """The API receives only observed frames; no lineage is an input feature."""
    import torch
    assert gap in (1,2,4,8) and len(images)==len(segs)>=2
    total=len(images);records=collections.defaultdict(list);emb=[];probes=[]
    # Arrays supplied here are normalized uint8 images, padded within this function.
    frames=[np.pad(x,24,mode='reflect') for x in images]
    for t,seg in enumerate(segs):
        yy,xx=np.nonzero(seg)
        if not len(yy):continue
        labs=seg[yy,xx];counts=np.bincount(labs);sy=np.bincount(labs,weights=yy);sx=np.bincount(labs,weights=xx)
        ids=np.flatnonzero(counts);cy=np.floor(sy[ids]/counts[ids]+.5).astype(np.int32);cx=np.floor(sx[ids]/counts[ids]+.5).astype(np.int32)
        for first in range(0,len(ids),128):
            last=first+128
            crops=np.stack([np.stack([frames[k][y:y+48,x:x+48] for k in context(t,total)]) for y,x in zip(cy[first:last],cx[first:last])])
            masks=crop_masks(seg,ids[first:last],cy[first:last],cx[first:last])
            with torch.inference_mode():
                z=encoder(torch.from_numpy(crops).to(device),torch.from_numpy(masks).to(device)).cpu().numpy().astype(np.float16)
            assert np.isfinite(z).all();emb.append(z)
            if first==0:probes.append(dict(t=t,seg_id=int(ids[0]),cy=int(cy[0]),cx=int(cx[0]),crop=crops[0].copy(),mask=masks[0].copy(),embedding=z[0].copy()))
        for k,v in [('t',np.full(len(ids),t)),('cy',cy),('cx',cx),('area',counts[ids]),('seg_id',ids)]:records[k].extend(map(int,v))
    assert emb,'No observed cells; preserve the failure'
    a={k:np.asarray(v,np.int32) for k,v in records.items()};a['embedding']=np.concatenate(emb)
    obs={k:a[k] for k in ['t','cy','cx','area']};ng=node_geometry(obs)
    ng=np.column_stack([ng,(a['t']==0).astype(float),(a['t']==total-1).astype(float),np.full(len(a['t']),np.log2(gap)/3)]).astype(np.float32)
    a['node_geometry']=ng.astype(np.float16);a['node_geometry_full']=ng
    return a,probes

def targets_for_view(a,segs,tras,track_rows,kept):
    projected=project_tracks(track_rows,kept);tracks={int(k):(int(b),int(e),int(p)) for k,b,e,p in projected}
    ids=np.zeros(len(a['t']),np.int32)
    for t,(seg,tra) in enumerate(zip(segs,tras)):
        mapping=identities(seg,tra);ii=np.flatnonzero(a['t']==t)
        ids[ii]=[mapping[int(k)] for k in a['seg_id'][ii]]
    obs={k:a[k] for k in ['t','cy','cx','area']};rows,audit=supervised(obs,ids,tracks)
    eg=np.zeros((len(rows['child']),11,15),np.float32);valid=rows['valid'].copy();valid[:,0]=False;rr,cc=np.nonzero(valid)
    if len(rr):eg[rr,cc]=edge_geometry(obs,a['node_geometry_full'],rows['parents'][rr,cc],rows['child'][rr])
    return dict(**rows,edge_geometry=eg.astype(np.float16),track_id=ids),audit,projected
