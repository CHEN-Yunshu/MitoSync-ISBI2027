"""Fixed 6000-update image-conditioned parent heads; no calibration read."""
import argparse,collections,hashlib,math,os,time
import numpy as np
import torch
from data_common import ROOT,HERE,read,put,sha
from dataset import Data,sample_digest_update
from model import ParentModel
ARMS=['Geo','R','RG','PM','Raw'];STEPS=6000

def runtime_check():
 seal=read(ROOT/'TRAINING_SEALED.json')
 for p,h in seal['sources'].items():assert sha(p)==h,p
 assert sha(ROOT/'PREPARED.json')==seal['prepared_sha256'];assert sha(ROOT/'PRETRAIN_VERIFIED.json')==seal['pretrain_sha256']
 assert sha(ROOT/'SAMPLING_EXPECTED.json')==seal['sampling_sha256']
 assert read(ROOT/'PRETRAIN_VERIFIED.json')['status']=='VERIFIED';return seal

def train(arm,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed);seal=runtime_check();d=ROOT/'models'/f'{arm}_s{seed}'
 assert not d.exists(),('Preserve prior/partial training',d);d.mkdir(parents=True)
 put(d/'STARTED.json',dict(arm=arm,seed=seed,time=time.time(),pid=os.getpid(),training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json')))
 data=Data();net=ParentModel();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.5*(1+math.cos(math.pi*s/STEPS)))
 digest=hashlib.sha256();losses=[];counts=collections.Counter();begin=time.time()
 for step in range(STEPS):
  tensors,samples,cc=data.batch(arm,rng);counts.update(cc);sample_digest_update(digest,samples)
  parent,child,edge,node,valid,target=tensors
  opt.zero_grad(set_to_none=True);out=net(parent,child,edge,node,valid);loss=torch.nn.functional.cross_entropy(out,target.long())
  assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.);assert torch.isfinite(norm)
  opt.step();scheduler.step()
  if step%100==0 or step==STEPS-1:
   r=dict(step=step+1,loss=float(loss.detach()),seconds=time.time()-begin)
   losses.append(r);tmp=d/'PROGRESS.tmp';tmp.write_text(__import__('json').dumps(r)+'\n');tmp.replace(d/'PROGRESS.json');print('TRAIN',arm,seed,r,flush=True)
 expected=read(ROOT/'SAMPLING_EXPECTED.json')[f'{arm}_s{seed}']
 assert digest.hexdigest()==expected['sample_digest'] and dict(counts)==expected['sample_counts'],'Training samples differ from independent plan'
 torch.save(dict(state_dict=net.state_dict(),arm=arm,seed=seed,updates=STEPS,training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json')),d/'model.pt')
 put(d/'DONE.json',dict(arm=arm,seed=seed,updates=STEPS,batch=64,seconds=time.time()-begin,finished_unix=time.time(),model_sha256=sha(d/'model.pt'),training_seal_sha256=sha(ROOT/'TRAINING_SEALED.json'),sample_digest=digest.hexdigest(),sample_counts=dict(counts),loss_samples=losses))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--seed',type=int,choices=range(1,6),required=True);a=ap.parse_args()
 try:train(a.arm,a.seed)
 except BaseException:
  import traceback
  put(ROOT/f'TRAINING_FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
