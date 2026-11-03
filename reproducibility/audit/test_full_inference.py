"""Synthetic contract tests through the archived inference and graph builders.

Run from an extracted v71-r2 package: python audit/test_full_inference.py.
No trained weights are used; this is not an accuracy experiment.
"""
from pathlib import Path
import importlib.util
import itertools
import sys
import types
import unittest
import warnings
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'audit/source'), str(ROOT/'reproducibility/source'), str(ROOT/'method')]
from inference import score_graph
from model import TripletModel
from decoder import decode

# Load the two unmodified, hash-matched upstream modules without importing the
# unrelated neural-network dependencies in the Trackastra package initializer.
prefix = 'archived_tracking'
pkg = types.ModuleType(prefix)
pkg.__path__ = [str(ROOT/'audit/upstream/trackastra/tracking')]
sys.modules[prefix] = pkg
for name in ['track_graph', 'tracking']:
    spec = importlib.util.spec_from_file_location(prefix+'.'+name, Path(pkg.__path__[0])/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
build_graph = sys.modules[prefix+'.tracking'].build_graph


def fixture(n=3):
    rng = np.random.default_rng(71)
    f = dict(t=np.r_[0,np.ones(n,dtype=int)], seg_id=np.arange(n+1)+4,
             area=rng.uniform(10,30,n+1), cy=rng.uniform(5,20,n+1), cx=rng.uniform(5,20,n+1),
             node_geometry=rng.normal(size=(n+1,6)).astype(np.float16),
             embedding=rng.normal(size=(n+1,256)).astype(np.float16))
    g = nx.DiGraph()
    for i in range(n+1): g.add_node(100*(i+1),time=int(f['t'][i]),label=int(f['seg_id'][i]))
    for i in range(n): g.add_edge(100,200+100*i,weight=.9-.07*i)
    torch.manual_seed(71)
    return g, f, TripletModel().eval()


def semantic_scores(g, scored):
    key = lambda n: (g.nodes[int(n)]['time'],g.nodes[int(n)]['label'])
    return {(key(p),tuple(sorted([key(a),key(b)]))):float(v)
            for p,a,b,v in zip(scored['mother'],scored['daughter_a'],scored['daughter_b'],scored['logit'])}


class FullInference(unittest.TestCase):
    def test_daughter_order_through_actual_inference(self):
        g,f,m = fixture(10)
        # Reverse canonical daughter order, while preserving cell features.
        relabel = {n:1300-n for n in g if n!=100}
        other = nx.relabel_nodes(g,relabel,copy=True)
        for arm in ['R','Geo']:
            a=semantic_scores(g,score_graph(g,m,arm,f,batch=7))
            b=semantic_scores(other,score_graph(other,m,arm,f,batch=7))
            self.assertEqual(set(a),set(b))
            np.testing.assert_allclose([a[k] for k in sorted(a)],[b[k] for k in sorted(a)],rtol=0,atol=1e-6)

    def test_feature_row_order_does_not_change_scores(self):
        g,f,m=fixture(); order=np.array([2,0,3,1])
        a=score_graph(g,m,'R',f);b=score_graph(g,m,'R',{k:v[order] for k,v in f.items()})
        np.testing.assert_array_equal(a['logit'],b['logit'])

    def test_all_inference_pairs_are_scored_without_training_pruning(self):
        g,f,m=fixture(10)
        self.assertEqual(len(score_graph(g,m,'R',f)['logit']),45)
        g.remove_edges_from(list(g.edges)[1:])
        self.assertEqual(len(score_graph(g,m,'R',f)['logit']),0)

    def test_invalid_time_and_duplicate_key_rejected(self):
        g,f,m=fixture();g.nodes[200]['time']=2;f['t'][1]=2
        with self.assertRaises(AssertionError):score_graph(g,m,'R',f)
        g,f,m=fixture();f['seg_id'][2]=f['seg_id'][1]
        with self.assertRaises(AssertionError):score_graph(g,m,'R',f)

    def test_decoding_semantics_survive_daughter_id_reversal(self):
        g,f,m=fixture();other=nx.relabel_nodes(g,{200:400,400:200},copy=True)
        solutions=[]
        for graph in [g,other]:
            scores=score_graph(graph,m,'R',f)
            pairs={tuple(map(int,k)):float(v) for k,v in zip(zip(scores['mother'],scores['daughter_a'],scores['daughter_b']),scores['residual'])}
            reference=graph.copy();reference.remove_edges_from(list(reference.edges))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected')
                sol,receipt=decode(graph,reference,{e:0. for e in graph.edges},pairs,beta=.25,alpha=0.)
            edges={(graph.nodes[p]['label'],graph.nodes[c]['label']) for p,c in sol.edges}
            solutions.append((edges,receipt['objective']))
        self.assertEqual(solutions[0][0],solutions[1][0])
        self.assertAlmostEqual(solutions[0][1],solutions[1][1],places=6)


class CandidateGraph(unittest.TestCase):
    def test_ten_nearest_available_successors_not_ten_closest_unfiltered(self):
        nodes=[dict(id=0,time=0,label=1,coords=(0.,0.))]
        nodes += [dict(id=i,time=1,label=i,coords=(0.,float(i))) for i in range(1,14)]
        weights=tuple(((0,i),.1+.01*i) for i in range(2,14))
        g=build_graph(nodes,weights,max_neighbors=10,max_distance=256,delta_t=1)
        self.assertEqual(set(g.successors(0)),set(range(2,12)))
        # Choice is spatial, not a ranking by association weight.
        self.assertNotIn(13,g.successors(0))

    def test_distance_inclusive_and_only_adjacent_observed_frames(self):
        nodes=[dict(id=i,time=t,label=i+1,coords=(0.,x)) for i,t,x in
               [(0,0,0.),(1,1,256.),(2,1,256.01),(3,2,1.)]]
        weights=tuple((e,.5) for e in [(0,1),(0,2),(0,3)])
        g=build_graph(nodes,weights,max_neighbors=10,max_distance=256,delta_t=1)
        self.assertEqual(set(g.edges),{(0,1)})
        self.assertEqual(len(g),4)

    def test_neighbor_cap_applies_per_parent_not_per_child(self):
        nodes=[dict(id=i,time=0,label=i+1,coords=(float(i),0.)) for i in range(12)]
        nodes.append(dict(id=12,time=1,label=13,coords=(0.,0.)))
        g=build_graph(nodes,tuple(((i,12),.5) for i in range(12)),max_neighbors=10,max_distance=256,delta_t=1)
        self.assertEqual(g.in_degree(12),12)


if __name__=='__main__':
    unittest.main(verbosity=2)
