import unittest
import numpy as np
import networkx as nx
from lfct_metrics import project,match_frame,prepare_mapping,counts

class Tests(unittest.TestCase):
    def gt(self):
        return dict(nodes={'1':dict(t=0,x=1,y=1),'2':dict(t=1,x=1,y=1),'3':dict(t=1,x=3,y=1)},edges=[['1','2'],['1','3']])
    def test_correct_exchange_and_missing_edge(self):
        a=np.zeros((5,5),int);a[1,1]=1;b=a.copy();b[1,3]=2
        p,m,_=prepare_mapping(self.gt(),[0,1],[a,b])
        g=nx.DiGraph();g.add_nodes_from([(8,dict(time=0,label=1)),(4,dict(time=1,label=1)),(2,dict(time=1,label=2))]);g.add_edges_from([(8,2),(8,4)])
        self.assertEqual(counts(g,p,m)['division_f1'],1)
        g.remove_edge(8,4);r=counts(g,p,m);self.assertEqual((r['tp'],r['fp'],r['fn']),(0,0,1))
    def test_outside_not_clipped_and_collision(self):
        a=np.ones((5,5),int)
        m,c=match_frame({'1':dict(x=-.1,y=2),'2':dict(x=2,y=2),'3':dict(x=1,y=1)},a)
        self.assertEqual(m,{'2':1});self.assertEqual(c['outside'],1);self.assertEqual(c['conflicts'],1)
    def test_sparse_projection_and_three_descendants(self):
        g=self.gt();g['nodes'].update({'4':dict(t=2,x=1,y=1),'5':dict(t=2,x=3,y=1)})
        g['edges'] += [['2','4'],['3','5']]
        p=project(g,[0,2]);self.assertEqual(p['events'],[('1',('4','5'))])
        g['nodes']['6']=dict(t=2,x=4,y=1);g['edges'].append(['3','6'])
        p=project(g,[0,2]);self.assertEqual(len(p['events']),0);self.assertEqual(len(p['nonbinary_branches']),1)
    def test_merge_rejected(self):
        g=self.gt();g['nodes']['4']=dict(t=0,x=4,y=4);g['edges'].append(['4','2'])
        with self.assertRaisesRegex(AssertionError,'Non-tree'):project(g,[0,1])
    def test_unmatched_daughter_is_false_negative(self):
        a=np.zeros((5,5),int);a[1,1]=1
        p,m,_=prepare_mapping(self.gt(),[0,1],[a,a])
        g=nx.DiGraph();g.add_nodes_from([(0,dict(time=0,label=1)),(1,dict(time=1,label=1))]);g.add_edge(0,1)
        self.assertEqual(counts(g,p,m)['fn'],1)
    def test_false_branch_without_gt(self):
        g=nx.DiGraph();g.add_nodes_from([(0,dict(time=0,label=1)),(1,dict(time=1,label=1)),(2,dict(time=1,label=2))]);g.add_edges_from([(0,1),(0,2)])
        p=dict(events=[],edges=[],nonbinary_branches=[])
        self.assertEqual(counts(g,p,{})['fp'],1)

if __name__=='__main__':unittest.main()
