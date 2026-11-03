"""Frozen CTC temporal tolerance and continuity controls. No training or selection."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,pickle,sys,time
import numpy as np
import networkx as nx
E=Path(os.environ.get('MITOSYNC_EVAL_ROOT','STUDY_ROOT/eval/mitosync'))
AB=E/'parent_ablation_20261029/v1';ROOT=E/'reviewer_resolution_20261103/ctc_tolerance_continuity_v1';FULL=E/'parent_fullview_20261024/v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(p):return pickle.loads(Path(p).read_bytes())
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
METRICS=['BC(0)','BC(1)','BC(2)','CT','TRA','DET','LNK','gt_divisions']+[f'{k}_div({j})' for j in range(3) for k in ['tp','fp','fn']]
def compact(row,job,arm):
 return {**{k:job[k] for k in ['dataset','gap','offset','seed']},'arm':arm,'metrics':{k:row['metrics'][k] for k in METRICS},'result_sha256':sha(row['_path'])}
def evaluate(item):
 jid,j=item;start=time.time();out=ROOT/'results'/jid
 if (out/'DONE.json').exists():return read(out/'RESULT.json')
 out.mkdir(parents=True,exist_ok=False)
 plan=read(AB/'PREPARED.json');seal=read(ROOT/'SEALED.json')
 for p in [j['graph'],j['triplets'],j['reference']]+j['masks']:assert sha(p)==plan['inputs'][p],p
 pp=AB/'results'/jid/'RESULT.json';old=read(pp);assert sha(pp)==read(AB/'VERIFIED.json')['result_hashes'][str(pp)]
 oldsol=pp.parent/'solution.pkl';assert sha(oldsol)==old['solution_sha256']
 dec=module('frozen_decoder',AB/'source/decoder.py');oracle=module('frozen_oracle',AB/'source/oracle.py')
 sys.path.insert(0,str(FULL/'evaluation_source'));import run as K
 V=module('export_verifier',FULL/'evaluation_source/verify.py')
 import tifffile
 from ctc_metrics import evaluate_sequence
 g=load(j['graph']);empty=nx.DiGraph();empty.add_nodes_from(g.nodes(data=True))
 with np.load(j['triplets']) as z:pairs={(int(p),int(a),int(b)):float(v) for p,a,b,v in zip(z['mother'],z['daughter_a'],z['daughter_b'],z['residual'])}
 ev={e:0. for e in g.edges};oldobj=dec.objective(g,list(load(oldsol).edges),ev,0.,pairs,j['beta'])
 sol,sr=dec.decode(g,empty,ev,pairs,j['beta'],0.);assert sr['objective']>=oldobj-1e-6
 opt=oracle.optimum(g,empty,ev,pairs,j['beta'],0.);assert abs(opt-sr['objective'])<1e-5
 assert set(sol)==set(g) and set(sol.edges)<=set(g.edges)
 masks=[tifffile.imread(p) for p in j['masks']];V.verify_gt(j['view'],Path(j['gt']))
 K.export(sol,np.stack(masks).astype(np.int32),out/'RES');V.verify_export(sol,out/'RES',masks)
 metrics=K.clean_metrics(evaluate_sequence(str(out/'RES'),j['gt'],metrics=['TRA','DET','LNK','BC','CT'],threads=1))
 assert metrics['gt_divisions']==old['metrics']['gt_divisions']
 for k in range(3):
  den=2*metrics[f'tp_div({k})']+metrics[f'fp_div({k})']+metrics[f'fn_div({k})'];assert abs(metrics[f'BC({k})']-2*metrics[f'tp_div({k})']/den)<1e-12
 for k in range(2):assert metrics[f'BC({k+1})']>=metrics[f'BC({k})']-1e-12
 with (out/'solution.pkl').open('xb') as f:pickle.dump(sol,f,protocol=4)
 put(out/'SOLVER.json',sr)
 row={**{k:j[k] for k in ['dataset','gap','offset','seed','beta']},'arm':'Pair '+j['arm']+' no continuity','metrics':{k:metrics[k] for k in METRICS},'seconds':time.time()-start,'solution_sha256':sha(out/'solution.pkl'),'solver_sha256':sha(out/'SOLVER.json'),'old_guarded_objective':oldobj,'independent_objective':opt,'input_hashes':{k:sha(p) for k,p in [('graph',j['graph']),('triplets',j['triplets']),('guarded_result',pp),('guarded_solution',oldsol)]}}
 put(out/'RESULT.json',row);put(out/'DONE.json',dict(status='VERIFIED',result_sha256=sha(out/'RESULT.json'),exports={p.name:sha(p) for p in (out/'RES').iterdir()}));return row

def summarize(rows):
 groups={}
 for ds in ['psc','fluo']:
  for gap in [1,8]:
   for arm in sorted({r['arm'] for r in rows}):
    rr=[r for r in rows if (r['dataset'],r['gap'],r['arm'])==(ds,gap,arm)]
    if not rr:continue
    assert len(rr)==5*gap and {(r['seed'],r['offset']) for r in rr}=={(s,o) for s in range(1,6) for o in range(gap)}
    groups[f'{ds}/g{gap}/{arm}']={}
    for metric in METRICS:
     values=[float(np.mean([r['metrics'][metric] for r in rr if r['seed']==s])) for s in range(1,6)]
     groups[f'{ds}/g{gap}/{arm}'][metric]=dict(values=values,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
 return groups

def run():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(AB/'PREPARED.json');verified=read(AB/'VERIFIED.json');assert verified['prepared_sha256']==sha(AB/'PREPARED.json')
 rows=[];seen=set();bindings={}
 for jid,j in plan['jobs'].items():
  pp=AB/'results'/jid/'RESULT.json';assert sha(pp)==verified['result_hashes'][str(pp)]
  for arm,p in [('Pair '+j['arm'],pp),('Full '+j['arm'],Path(j['accepted'])),('Graph',Path(j['reference']).parent/'RESULT.json')]:
   key=(j['dataset'],j['gap'],j['offset'],j['seed'],arm)
   if key in seen:continue
   seen.add(key);row=read(p);row['_path']=str(p);rows.append(compact(row,j,arm));bindings[str(p)]=sha(p)
   assert sha(p.parent/'solution.pkl')==row['solution_sha256']
   if arm.startswith('Full'):assert sha(p)==plan['inputs'][str(p)]
 assert len(rows)==450
 if not (ROOT/'HISTORICAL_TOLERANCE.json').exists():put(ROOT/'HISTORICAL_TOLERANCE.json',dict(scope='Existing recorded predictions; no rerun',groups=summarize(rows),rows=rows,bindings=bindings))
 jobs=sorted(plan['jobs'].items());assert len(jobs)==180
 if not (ROOT/'SEALED.json').exists():put(ROOT/'SEALED.json',dict(protocol='All 180 fixed Pair R/RG CTC views, five seeds, native and all eight offsets; remove continuity only, alpha=0, original beta/candidates/scores. BC(0/1/2), CT, TRA, DET, LNK. Exploratory descriptive reporting; no model or threshold selection.',source_sha256=sha(__file__),prepared_sha256=sha(AB/'PREPARED.json'),verified_sha256=sha(AB/'VERIFIED.json'),decoder_sha256=sha(AB/'source/decoder.py'),oracle_sha256=sha(AB/'source/oracle.py'),workers=2,gpu_workers=0))
 else:assert read(ROOT/'SEALED.json')['source_sha256']==sha(__file__)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(evaluate,jobs),1):
   rows.append(row);print(json.dumps(dict(completed=i,total=180,last=[row['dataset'],row['gap'],row['offset'],row['seed'],row['arm']])),flush=True)
 put(ROOT/'SUMMARY.json',dict(groups=summarize(rows),rows=rows,scope='Exploratory fixed-score controls; eight offsets within seed, then five seed mean/SD; not a newly held-out test.'))
 put(ROOT/'VERIFIED.json',dict(status='VERIFIED',historical_views=450,new_decodes=180,summary_sha256=sha(ROOT/'SUMMARY.json'),historical_sha256=sha(ROOT/'HISTORICAL_TOLERANCE.json'),sealed_sha256=sha(ROOT/'SEALED.json')))
 put(ROOT/'DONE.json',dict(status='DONE',verified_sha256=sha(ROOT/'VERIFIED.json'),time=time.time()))
if __name__=='__main__':run()
