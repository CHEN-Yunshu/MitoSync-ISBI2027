"""Frozen real-only encoder; actual cell mask conditions spatial pooling."""
import numpy as np
import torch
import torch.nn.functional as F

class Encoder(torch.nn.Module):
 def __init__(self,checkpoint):
  super().__init__()
  from torchvision.models import resnet18
  m=resnet18(weights=None);m.fc=torch.nn.Linear(512,1)
  m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
  self.trunk=torch.nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2)
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
  self.requires_grad_(False);self.eval()
 def forward(self,crops,masks):
  assert not self.training and crops.ndim==4 and crops.shape[1:]==(3,48,48) and masks.shape==crops[:,0].shape
  x=(crops.float()/255.-self.mean)/self.std
  x=F.interpolate(x,size=(96,96),mode='bilinear',align_corners=False);z=self.trunk(x)
  roi=F.interpolate(masks[:,None].float(),size=z.shape[-2:],mode='area')
  def pool(w):return F.normalize((z*w).sum((2,3))/w.sum((2,3)).clamp_min(1e-6),dim=1,eps=1e-8)
  return torch.cat([pool(roi),pool(1-roi)],dim=1)

def crop_masks(seg,labels,ys,xs):
 """Outside the image is background; no reflected duplicate identities."""
 pad=np.pad(seg,24,mode='constant')
 out=np.stack([pad[int(y):int(y)+48,int(x):int(x)+48]==int(k) for k,y,x in zip(labels,ys,xs)])
 assert out.shape==(len(labels),48,48) and np.all(out.sum((1,2))>0)
 return out
