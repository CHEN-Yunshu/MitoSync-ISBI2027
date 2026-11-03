"""Run: python reproducibility/test_input_contract.py.

Requires NumPy, SciPy, PyTorch and NetworkX. Uses synthetic fixtures only, not
trained weights, and does not estimate real-data accuracy.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import networkx as nx

sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from features import context, view_features
from triplets import descriptors
from model import TripletModel
from lfct_metrics import counts, match_frame
from test_triplets import TripleTests
from test_lfct_metrics import Tests


class InputContract(unittest.TestCase):
    def test_all_inputs_exchange_invariant(self):
        rng=np.random.default_rng(71); n=100
        data=dict(area=rng.uniform(2,200,n),cy=rng.uniform(0,200,n),cx=rng.uniform(0,200,n),
                  node_geometry=rng.normal(size=(n,6)).astype(np.float16))
        idx=np.array([rng.choice(n,3,replace=False) for _ in range(127)])
        a=descriptors(data,idx); b=descriptors(data,idx[:,[0,2,1]])
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
        torch.manual_seed(71); model=TripletModel().eval(); emb=torch.randn(n,256)
        with torch.inference_mode():
            x=model(emb[idx[:,0]],emb[idx[:,1]],emb[idx[:,2]],torch.from_numpy(a))
            y=model(emb[idx[:,0]],emb[idx[:,2]],emb[idx[:,1]],torch.from_numpy(b))
        torch.testing.assert_close(x,y,rtol=0,atol=1e-6)

    def test_sparse_crops_use_observed_frames_and_repeat_endpoints(self):
        # Values encode original frame IDs. Hidden frames are never passed in.
        images=[np.full((64,64),i,np.uint8) for i in [3,11,19,27]]
        masks=[]
        for _ in images:
            m=np.zeros((64,64),np.int32);m[30:34,30:34]=1;masks.append(m)
        class Capture:
            def __init__(self):self.seen=[]
            def __call__(self,crops,mask):
                self.seen.extend(crops[:, :, 0, 0].tolist())
                return torch.zeros(len(crops),256)
        encoder=Capture(); data,_=view_features(images,masks,encoder,8)
        self.assertEqual(encoder.seen,[[3,3,11],[3,11,19],[11,19,27],[19,27,27]])
        np.testing.assert_array_equal(data['node_geometry_full'][:,3:5],[[1,0],[0,0],[0,0],[0,1]])
        np.testing.assert_array_equal(data['node_geometry_full'][:,5],np.ones(4))

    def test_no_event_scoring_and_deterministic_tie(self):
        p=dict(events=[],edges=[],nonbinary_branches=[])
        g=nx.DiGraph()
        self.assertIsNone(counts(g,p,{})['division_f1'])
        for i,t in [(0,0),(1,1),(2,1)]:g.add_node(i,time=t,label=i+1)
        g.add_edges_from([(0,1),(0,2)])
        self.assertEqual(counts(g,p,{})['division_f1'],0)
        mask=np.ones((5,5),int)
        m,a=match_frame({'9':dict(x=1,y=2),'2':dict(x=3,y=2)},mask)
        self.assertEqual(m,{'2':1});self.assertEqual(a['conflicts'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
