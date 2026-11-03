"""Training-only daughter-pair targets; inference descriptors never use identities."""
from collections import defaultdict
from itertools import combinations
import numpy as np

def build(data,views=None):
    known=None if views is None else [{int(r[0]) for r in v['projected_tracks']} for v in views]
    groups=defaultdict(list);true=defaultdict(list)
    for row,c in enumerate(data['child']):
        c=int(c);target=int(data['target'][row]);kind=int(data['kind'][row])
        for col in np.flatnonzero(data['valid'][row])[1:]:
            p=int(data['parents'][row,col]);assert p>=0
            assert data['view_id'][p]==data['view_id'][c] and data['t'][p]+1==data['t'][c]
            distance=float(data['edge_geometry'][row,col,13])
            groups[p].append((distance,c,int(data['gap'][row])))
            if col==target and kind==1:true[p].append(c)
    rows=[]
    for p,children in sorted(groups.items()):
        ident=int(data['track_id'][p])
        if ident<=0 or (known is not None and ident not in known[int(data['view_id'][p])]):continue
        assert len({c for _,c,_ in children})==len(children)
        assert len({g for _,_,g in children})==1
        gap=children[0][2];positive=set(combinations(sorted(true[p]),2))
        # Closest-four negatives are outcome-independent hard candidates.
        # Add all positive pairs for training coverage; never at inference.
        near=sorted(c for _,c,_ in sorted(children)[:4])
        pairs=set(combinations(near,2))|positive
        for a,b in sorted(pairs):rows.append((p,a,b,int((a,b) in positive),gap))
    return np.asarray(rows,dtype=np.int32).reshape(-1,5)

def verify_targets(data,views,rows):
    tracks=[{int(k):(int(b),int(e),int(p)) for k,b,e,p in v['projected_tracks']} for v in views]
    for p,a,b,y,gap in rows:
        v=int(data['view_id'][p]);assert data['view_id'][a]==data['view_id'][b]==v and views[v]['gap']==gap
        t=int(data['t'][p]);assert data['t'][a]==data['t'][b]==t+1 and a<b
        pm,ia,ib=map(int,data['track_id'][[p,a,b]]);tr=tracks[v]
        assert pm in tr and ia in tr and ib in tr and ia!=ib
        expected=int(tr[ia][2]==pm and tr[ib][2]==pm and tr[ia][0]==tr[ib][0]==t+1 and tr[pm][1]==t)
        assert y==expected,(p,a,b,y,expected)
    assert len(rows)==len(set(map(tuple,rows[:,:3])))

def descriptors(data,indices):
    """Symmetric observed-only geometry: mother and an unordered daughter pair."""
    idx=np.asarray(indices,dtype=np.int64);assert idx.ndim==2 and idx.shape[1]==3
    p,a,b=idx.T;area=np.asarray(data['area'],dtype=np.float64)
    xy=np.column_stack([data['cy'],data['cx']]).astype(np.float64)
    ap,aa,ab=area[p],area[a],area[b];assert min(ap.min(),aa.min(),ab.min())>0
    rad=np.sqrt(ap/np.pi);da=xy[a]-xy[p];db=xy[b]-xy[p]
    na=np.linalg.norm(da,axis=1);nb=np.linalg.norm(db,axis=1)
    mid=(aa[:,None]*da+ab[:,None]*db)/(aa+ab)[:,None]
    ng=np.asarray(data['node_geometry'],np.float32)
    geom=np.column_stack([np.log((aa+ab)/ap),abs(np.log(aa/ab)),np.linalg.norm(mid,axis=1)/rad,
        np.linalg.norm(xy[a]-xy[b],axis=1)/rad,np.minimum(na,nb)/rad,np.maximum(na,nb)/rad,
        np.sum(da*db,axis=1)/np.maximum(na*nb,1e-8),ng[p],(ng[a]+ng[b])/2,abs(ng[a]-ng[b])]).astype(np.float32)
    assert geom.shape==(len(idx),25) and np.isfinite(geom).all()
    return np.clip(geom,-12,12)
