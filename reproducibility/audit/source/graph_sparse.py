"""Observed-only sparse PSC02 development graphs, no test video access."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,pickle,subprocess,sys,time,traceback
from pathlib import Path
N=Path('STUDY_ROOT');E=N/'eval/mitosync';ROOT=E/'rate_adaptation_20261025/v1';BASE=E/'event_association_20261024/v1';TAIL=N/'data/ctc/PhC-C2DL-PSC_valtail';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))
import pilot as P

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)

def prepare():
 inputs={};models={}
 for b in ['ChHead','R18','OURS']:
  for s in range(1,6):
   d=P.model_dir(b,s);models[f'{b}_s{s}']=str(d)
   for n in ['model.pt','config.yaml','train_config.yaml']:inputs[str(d/n)]=sha(d/n)
 images=sorted((TAIL/'02').glob('t*.tif'));assert len(images)==45
 for sub,pat in [('02','t*.tif'),('02_ST/SEG','man_seg*.tif'),('02_GT/TRA','man_track*')]:
  for p in sorted((TAIL/sub).glob(pat)):inputs[str(p)]=sha(p)
 import trackastra
 sources={str(p):sha(p) for p in Path(trackastra.__file__).parent.rglob('*.py')}
 sources[str(HERE/'prepare_graphs.py')]=sha(HERE/'prepare_graphs.py')
 put(ROOT/'PREPARED.json',dict(time=time.time(),inputs=inputs,sources=sources,models=models,gap=8,offsets=list(range(8)),seeds=list(range(1,6)),graphs=120,scope='Historically exposed PSC02 tail development; 45 observed-source frames; no 01',protocol_sha256=sha(HERE/'PROTOCOL.md')))

def worker(b,s):
 import torch,numpy as np,tifffile
 from trackastra.model import Trackastra
 from trackastra.tracking import build_graph
 torch.set_num_threads(1);torch.manual_seed(20261025)
 seal=read(ROOT/'PREPARED.json');folder=Path(seal['models'][f'{b}_s{s}'])
 for n in ['model.pt','config.yaml','train_config.yaml']:assert sha(folder/n)==seal['inputs'][str(folder/n)]
 model=Trackastra.from_folder(folder,device='cpu')
 for offset in range(8):
  d=ROOT/'graphs'/f'g8_o{offset}'/f'{b}_s{s}';d.mkdir(parents=True,exist_ok=False);begin=time.time()
  kept=list(range(offset,45,8));im=[];ma=[]
  for t in kept:
   ip=TAIL/'02'/f't{t:03d}.tif';mp=TAIL/'02_ST/SEG'/f'man_seg{t:03d}.tif'
   for p in [ip,mp]:assert sha(p)==seal['inputs'][str(p)]
   im.append(tifffile.imread(ip));ma.append(tifffile.imread(mp))
  imgs=np.stack(im);masks=np.stack(ma).astype(np.int32)
  pred=model._predict(imgs,masks,edge_threshold=.05,n_workers=0,batch_size=1)
  graph=build_graph(pred['nodes'],pred['weights'],max_distance=256,max_neighbors=10,delta_t=1)
  areas=[np.bincount(m.ravel()) for m in masks]
  for n,v in graph.nodes(data=True):v['area']=int(areas[v['time']][v['label']])
  with (d/'graph.pkl').open('xb') as f:pickle.dump(graph,f,protocol=4)
  put(d/'DONE.json',dict(time=time.time(),seconds=time.time()-begin,backbone=b,seed=s,offset=offset,source_frames=kept,graph_sha256=sha(d/'graph.pkl'),model_sha256=sha(folder/'model.pt'),prepared_sha256=sha(ROOT/'PREPARED.json'),nodes=len(graph),edges=graph.number_of_edges(),device='cpu',hidden_native_frames_not_supplied=True))
  print('GRAPH',b,s,offset,time.time()-begin,flush=True)

def task(b,s):
 with (ROOT/f'{b}_s{s}.log').open('x') as log:subprocess.run([sys.executable,'-u',str(HERE/'prepare_graphs.py'),'worker','--backbone',b,'--seed',str(s)],stdout=log,stderr=subprocess.STDOUT,check=True)
def driver():
 lock=(ROOT/'DRIVER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
 # One real model/view smoke group before parallel dispatch catches runtime failures early.
 task('ChHead',1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  fs=[pool.submit(task,b,s) for b in ['ChHead','R18','OURS'] for s in range(1,6) if (b,s)!=('ChHead',1)]
  for f in concurrent.futures.as_completed(fs):f.result()
 receipts=list((ROOT/'graphs').glob('*/*/DONE.json'));assert len(receipts)==120
 for p in receipts:assert sha(p.parent/'graph.pkl')==read(p)['graph_sha256']
 for p,h in {**read(ROOT/'PREPARED.json')['inputs'],**read(ROOT/'PREPARED.json')['sources']}.items():assert sha(p)==h,p
 put(ROOT/'GRAPHS_READY.json',dict(time=time.time(),graphs=120,receipts={str(p):sha(p) for p in receipts},performance_not_yet_evaluated=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['driver','worker']);ap.add_argument('--backbone');ap.add_argument('--seed',type=int);a=ap.parse_args()
 try:
  if a.mode=='driver':driver()
  else:worker(a.backbone,a.seed)
 except BaseException:
  put(ROOT/f'FAILURE_{os.getpid()}_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
