"""Unordered daughter-pair evidence conditioned on the proposed mother."""
import torch
from torch import nn

class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(256,32),nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(217,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,mother,a,b,geometry):
        m=self.project(mother);a=self.project(a);b=self.project(b);mean=(a+b)/2
        features=torch.cat([m,mean,abs(a-b),a*b,abs(m-mean),m*mean,geometry],dim=-1)
        return self.classifier(features).squeeze(-1)
