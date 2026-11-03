import unittest
import numpy as np
from triplets import build,verify_targets,descriptors

class TripleTests(unittest.TestCase):
    def fixture(self):
        return dict(child=np.array([2,3,4]),target=np.array([1,1,2]),kind=np.array([1,1,0]),
            parents=np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),valid=np.ones((3,3),bool),
            edge_geometry=np.zeros((3,3,15)),view_id=np.zeros(5,int),t=np.array([0,0,1,1,1]),gap=np.ones(3,int),
            track_id=np.array([1,2,3,4,2]),area=np.array([20,20,9,11,20]),cy=np.array([10,30,9,11,30]),cx=np.array([10,30,8,12,31]),node_geometry=np.zeros((5,6)))
    def test_wrong_neighbor_pair_is_negative(self):
        d=self.fixture();rows=build(d);views=[dict(gap=1,projected_tracks=[[1,0,0,0],[2,0,1,0],[3,1,1,1],[4,1,1,1]])]
        verify_targets(d,views,rows)
        self.assertEqual([tuple(r) for r in rows if r[3]],[(0,2,3,1,1)])
        self.assertIn((0,2,4,0,1),list(map(tuple,rows)))
    def test_observed_descriptor_symmetry(self):
        d=self.fixture();idx=np.array([[0,2,3],[1,2,4]])
        np.testing.assert_array_equal(descriptors(d,idx),descriptors(d,idx[:,[0,2,1]]))
        x=descriptors(d,idx);d['track_id'][:]=-999;d['target'][:]=0
        np.testing.assert_array_equal(x,descriptors(d,idx))
    def test_network_symmetry(self):
        import torch
        from model import TripletModel
        torch.manual_seed(18);m=TripletModel().eval();x=torch.randn(7,3,256);g=torch.randn(7,25)
        with torch.inference_mode():
            a=m(x[:,0],x[:,1],x[:,2],g);b=m(x[:,0],x[:,2],x[:,1],g)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
