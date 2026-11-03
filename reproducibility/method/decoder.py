"""Parent-identity residuals in the fixed, continuity-guarded cap-three decoder."""
import itertools, time
import networkx as nx
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def objective(graph, edges, evidence, alpha, pairs, beta):
    out=nx.DiGraph();out.add_nodes_from(graph);out.add_edges_from(edges)
    return sum(float(graph[p][c]['weight'])+alpha*evidence[(p,c)] for p,c in edges) + sum(.75*(2-out.out_degree(p)) for p in out if out.out_degree(p)) + beta*sum(float(np.mean([pairs[(p,a,b)] for a,b in itertools.combinations(sorted(out.successors(p)),2)])) for p in out if out.out_degree(p)>=2)


def decode(graph, reference, evidence, pairs, beta, alpha=.25):
    assert alpha in (0., .25)
    assert beta in (0.,.125,.25,.5)
    expected={(p,a,b) for p in graph for a,b in itertools.combinations(sorted(graph.successors(p)),2)}
    assert set(pairs)==expected and all(np.isfinite(v) and -4<=v<=4 for v in pairs.values())
    assert set(graph)==set(reference) and set(reference.edges)<=set(graph.edges)
    assert set(evidence)==set(graph.edges) and alpha>=0
    assert all(np.isfinite(v) and -4<=v<=4 for v in evidence.values())
    assert all(reference.in_degree(n)<=1 and reference.out_degree(n)<=3 for n in reference)
    assert all(graph.nodes[c]['time']==graph.nodes[p]['time']+1 for p,c in graph.edges)
    out=nx.DiGraph();out.add_nodes_from(graph.nodes(data=True));receipts=[]
    for t in sorted({v['time'] for _,v in graph.nodes(data=True)}):
        parents=sorted(n for n,v in graph.nodes(data=True) if v['time']==t);options=[];values=[]
        for p in parents:
            children=sorted(graph.successors(p))
            if reference.out_degree(p)==0:options.append((p,()));values.append(0.)
            for size in range(1,min(3,len(children))+1):
                for cs in itertools.combinations(children,size):
                    options.append((p,cs));values.append(sum(float(graph[p][c]['weight'])+alpha*evidence[(p,c)] for c in cs)+.75*(2-size)+(beta*float(np.mean([pairs[(p,a,b)] for a,b in itertools.combinations(cs,2)])) if size>=2 else 0.))
        children=sorted({c for _,cs in options for c in cs});pi={p:i for i,p in enumerate(parents)};ci={c:i+len(parents) for i,c in enumerate(children)}
        rows=[];cols=[]
        for i,(p,cs) in enumerate(options):
            rows.append(pi[p]);cols.append(i)
            for c in cs:rows.append(ci[c]);cols.append(i)
        matrix=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(parents)+len(children),len(options))).tocsc()
        lower=np.r_[np.ones(len(parents)),np.zeros(len(children))];begin=time.time()
        result=milp(-np.asarray(values),integrality=np.ones(len(options)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lower,1),options={'time_limit':120.,'mip_rel_gap':0.,'threads':1})
        assert result.success and result.status==0,(result.status,result.message)
        assert np.max(abs(result.x-np.round(result.x)))<1e-6
        selected=np.round(result.x);assert np.all(matrix@selected>=lower-1e-8) and np.all(matrix@selected<=1+1e-8)
        for i,(p,cs) in enumerate(options):
            if selected[i]:out.add_edges_from((p,c,graph[p][c]) for c in cs)
        receipts.append(dict(frame=int(t),objective=float(np.asarray(values)@selected),options=len(options),seconds=time.time()-begin))
    assert all(out.in_degree(n)<=1 and out.out_degree(n)<=3 for n in out)
    assert all(out.out_degree(p)>0 for p in reference if reference.out_degree(p)>0)
    return out,dict(alpha=alpha,beta=beta,reference_replay=False,frames=receipts,objective=objective(graph,list(out.edges),evidence,alpha,pairs,beta))
