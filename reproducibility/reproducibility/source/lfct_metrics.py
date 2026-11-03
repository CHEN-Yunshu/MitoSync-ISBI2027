"""Point-annotated lineage evaluation, independent of all learned predictions.

This is centroid-in-mask division/edge evaluation, not CTC mask-IoU metrics.
Out-of-frame points are retained as unmatched. No nearest-cell substitution.
"""
from collections import defaultdict, Counter
import math
import numpy as np

def project(gt, kept):
    nodes=gt['nodes'];kept=list(map(int,kept));index={t:i for i,t in enumerate(kept)}
    assert len(index)==len(kept) and all(a<b for a,b in zip(kept,kept[1:]))
    parents=defaultdict(list);edges=set()
    for a,b in gt['edges']:
        assert a in nodes and b in nodes and nodes[a]['t']<nodes[b]['t']
        assert (a,b) not in edges,'Duplicate ground-truth edge'
        edges.add((a,b));parents[b].append(a)
    assert all(len(p)==1 for p in parents.values()),'Non-tree annotation requires adjudication; do not select a parent'
    retained={k:n for k,n in nodes.items() if n['t'] in index};new=[];disconnected=0
    for c,n in retained.items():
        i=index[n['t']]
        if i==0:continue
        target=kept[i-1];p=parents[c][0] if parents[c] else None
        while p is not None and nodes[p]['t']>target:p=parents[p][0] if parents[p] else None
        if p is not None and nodes[p]['t']==target:new.append((p,c))
        elif parents[c]:disconnected+=1
    child=defaultdict(list)
    for a,b in new:child[a].append(b)
    events=[(p,tuple(sorted(cs))) for p,cs in child.items() if len(cs)==2]
    return dict(nodes=retained,edges=new,events=events,kept=kept,
        nonbinary_branches=[(p,cs) for p,cs in child.items() if len(cs)>2],
        annotation_gaps_or_absent_observed_ancestor=disconnected)

def match_frame(nodes,mask):
    """Maximum-cardinality matching when a point has one containing mask.

    Floating coordinates must lie within the original image. Nearest pixel uses
    floor(x+0.5); coordinates rounding outside the image remain unmatched.
    """
    h,w=mask.shape;groups=defaultdict(list);outside=0
    for k,n in nodes.items():
        x,y=float(n['x']),float(n['y'])
        assert math.isfinite(x) and math.isfinite(y)
        ix,iy=math.floor(x+.5),math.floor(y+.5)
        if not (0<=x<w and 0<=y<h and 0<=ix<w and 0<=iy<h):outside+=1;continue
        lab=int(mask[iy,ix])
        if lab:groups[lab].append((k,n))
    yy,xx=np.nonzero(mask);labs=mask[yy,xx];counts=np.bincount(labs)
    sy=np.bincount(labs,weights=yy);sx=np.bincount(labs,weights=xx);out={}
    for lab,items in groups.items():
        cy,cx=sy[lab]/counts[lab],sx[lab]/counts[lab]
        k,_=min(items,key=lambda kv:((kv[1]['y']-cy)**2+(kv[1]['x']-cx)**2,int(kv[0])))
        out[k]=lab
    return out,dict(outside=outside,conflicts=sum(len(x)-1 for x in groups.values()),matched=len(out),total=len(nodes))

def prepare_mapping(gt,kept,masks):
    p=project(gt,kept);byframe=defaultdict(dict)
    for k,n in p['nodes'].items():byframe[n['t']][k]=n
    mapping={};audit=Counter()
    for t,(original,mask) in enumerate(zip(kept,masks)):
        m,a=match_frame(byframe[original],mask);audit.update(a)
        for k,lab in m.items():mapping[k]=(t,lab)
    assert len(mapping)==len(set(mapping.values()))
    return p,mapping,dict(audit)

def counts(solution,p,mapping):
    lookup={(int(v['time']),int(v['label'])):n for n,v in solution.nodes(data=True)}
    assert len(lookup)==len(solution)
    matches={k:lookup[tl] for k,tl in mapping.items() if tl in lookup}
    pred={n:tuple(sorted(solution.successors(n))) for n in solution if solution.out_degree(n)>=2}
    assert all(solution.in_degree(n)<=1 for n in solution)
    found=set();ceiling=0
    for m,ds in p['events']:
        if m not in matches or any(c not in matches for c in ds):continue
        a=matches[m];cs=tuple(sorted(matches[c] for c in ds));assert len(set(cs))==2
        ceiling+=1
        if a in pred and pred[a]==cs:found.add(a)
    tp=len(found);fp=len(pred)-tp;fn=len(p['events'])-tp;den=2*tp+fp+fn
    expected={(matches[a],matches[b]) for a,b in p['edges'] if a in matches and b in matches}
    ep=set(solution.edges);etp=len(ep&expected);efp=len(ep)-etp;efn=len(p['edges'])-etp;eden=2*etp+efp+efn
    return dict(tp=tp,fp=fp,fn=fn,division_f1=2*tp/den if den else None,
        gt_divisions=len(p['events']),predicted_divisions=len(pred),matched_endpoint_ceiling=ceiling,
        excluded_nonbinary_gt_branches=len(p['nonbinary_branches']),
        edge_tp=etp,edge_fp=efp,edge_fn=efn,edge_f1=2*etp/eden if eden else None,
        gt_edges=len(p['edges']),predicted_edges=len(ep))
