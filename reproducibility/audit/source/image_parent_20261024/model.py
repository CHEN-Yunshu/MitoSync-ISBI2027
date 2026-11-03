"""Shared ROI projection and explicit parent/null logits."""
import torch
from torch import nn

class ParentModel(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Sequential(nn.Linear(256,64),nn.GELU())
  self.edge=nn.Sequential(nn.Linear(265,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
  self.null=nn.Sequential(nn.Linear(67,64),nn.GELU(),nn.Linear(64,1))
 def edge_logits(self,p,c,geometry):
  a=self.project(p);b=self.project(c)
  return self.edge(torch.cat([a,b,abs(a-b),a*b,geometry],dim=-1)).squeeze(-1)
 def null_logits(self,c,geometry):return self.null(torch.cat([self.project(c),geometry],dim=-1)).squeeze(-1)
 def forward(self,parents,children,edge_geometry,node_geometry,valid):
  n,k,d=parents.shape
  edges=self.edge_logits(parents[:,1:],children[:,None,:].expand(-1,k-1,-1),edge_geometry[:,1:])
  null=self.null_logits(children,node_geometry)
  return torch.cat([null[:,None],edges],dim=1).masked_fill(~valid,-1e9)
