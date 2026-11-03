"""Independent XML, pixel-assignment, graph and metric readback for frozen LFCT.

Does not import the production metric module. Outcomes cannot change inclusion,
models or settings. This file is sealed separately before aggregate scores read.
"""
from pathlib import Path
from collections import defaultdict,Counter
import itertools,json,hashlib,math,pickle,time,sys,xml.etree.ElementTree as ET
import numpy as np
import tifffile
E=Path('STUDY_ROOT/eval/mitosync')
ROOT=E/'reviewer_sprint_20261101/lfct_v1'
UP=Path('EXTERNAL_DATA_ROOT')
AUD=ROOT/'independent_v2'
ARMS=['TrackastraGraph','TrackastraGreedy','R','RG','ProgramPaintMatched','RenderedMatched']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def check(v):
 for p,h in v.items():assert sha(p)==h,p

def xml_truth(p):
 root=ET.parse(p).getroot();info=root.find('.//ImageData').attrib
 s={x.get('ID'):x for x in root.findall('.//AllSpots/SpotsInFrame/Spot')}
 assert len(s)==len(root.findall('.//AllSpots/SpotsInFrame/Spot'))
 approved={x.get('TRACK_ID') for x in root.findall('.//FilteredTracks/TrackID')};edges=[]
 for track in root.findall('.//AllTracks/Track'):
  if approved and track.get('TRACK_ID') not in approved:continue
  for x in track.findall('Edge'):
   a,b=x.get('SPOT_SOURCE_ID'),x.get('SPOT_TARGET_ID')
   a,b=sorted([a,b],key=lambda k:float(s[k].get('FRAME')))
   assert float(s[a].get('FRAME'))<float(s[b].get('FRAME'));edges.append((a,b))
 assert len(edges)==len(set(edges));active=set(itertools.chain.from_iterable(edges));nodes={}
 for k,x in s.items():
  if k not in active and float(x.get('VISIBILITY','1'))==0:continue
  t=float(x.get('FRAME'));assert t.is_integer()
  nodes[k]=dict(t=int(t),x=float(x.get('POSITION_X'))/float(info['pixelwidth']),y=float(x.get('POSITION_Y'))/float(info['pixelheight']))
 return nodes,set(edges)

def audit_case(case):
 w=ROOT/'cases'/case;receipt=read(w/'ADMISSION.json');ready=read(w/'input_ready.json')
 check(receipt['bindings']);assert sha(w/'ground_truth.json')==receipt['ground_truth_sha256']
 xml=UP/'raw'/case/'annotations.xml';assert sha(xml)==receipt['annotation_sha256']
 nodes,edges=xml_truth(xml);gt=read(w/'ground_truth.json');assert set(nodes)==set(gt['nodes']) and edges==set(map(tuple,gt['edges']))
 for k,n in nodes.items():assert n=={v:gt['nodes'][k][v] for v in ['t','x','y']}
 indeg=Counter(b for a,b in edges);assert receipt['held']==any(x>1 for x in indeg.values())
 if not receipt['held']:
  mr=read(w/'MASKS_READY.json');check(mr['bindings'])
  assert len(mr['paths'])==ready['frames']
  for i,p in enumerate(mr['paths']):
   mask=tifffile.imread(p);assert mask.shape==(ready['height'],ready['width']) and mask.dtype.kind in 'ui' and mask.min()>=0
 return dict(case=case,nodes=len(nodes),edges=len(edges),held=receipt['held'],admission_sha256=sha(w/'ADMISSION.json'),xml_sha256=sha(xml),mask_receipt_sha256=sha(w/'MASKS_READY.json') if not receipt['held'] else None)

def independent_mapping(case,kept):
 w=ROOT/'cases'/case;nodes,edges=xml_truth(UP/'raw'/case/'annotations.xml');parents={b:a for a,b in edges}
 assert len(parents)==len(edges)
 retained={k:n for k,n in nodes.items() if n['t'] in kept};index={t:i for i,t in enumerate(kept)};projected=[];gaps=0
 for k,n in retained.items():
  i=index[n['t']]
  if i==0:continue
  p=parents.get(k)
  while p is not None and nodes[p]['t']>kept[i-1]:p=parents.get(p)
  if p is not None and nodes[p]['t']==kept[i-1]:projected.append((p,k))
  elif k in parents:gaps+=1
 mr=read(w/'MASKS_READY.json');mapping={};audit=Counter()
 for ti,t in enumerate(kept):
  mask=tifffile.imread(mr['paths'][t]);h,ww=mask.shape;claims=defaultdict(list)
  for k,n in retained.items():
   if n['t']!=t:continue
   audit['total']+=1;x,y=n['x'],n['y'];ix,iy=int(math.floor(x+.5)),int(math.floor(y+.5))
   if not (0<=x<ww and 0<=y<h and 0<=ix<ww and 0<=iy<h):audit['outside']+=1;continue
   lab=int(mask[iy,ix])
   if lab:claims[lab].append(k)
  from scipy.ndimage import center_of_mass
  centroids=dict(zip(claims,center_of_mass(np.ones(mask.shape,dtype=np.uint8),mask,list(claims))))
  for lab,ks in claims.items():
   cy,cx=centroids[lab]
   k=sorted(ks,key=lambda k:((nodes[k]['x']-cx)**2+(nodes[k]['y']-cy)**2,int(k)))[0]
   mapping[k]=(ti,lab);audit['matched']+=1;audit['conflicts']+=len(ks)-1
 children=defaultdict(list)
 for a,b in projected:children[a].append(b)
 events={a:frozenset(bs) for a,bs in children.items() if len(bs)==2}
 nonbinary={a:bs for a,bs in children.items() if len(bs)>2}
 return retained,set(projected),events,nonbinary,mapping,dict(audit),gaps

def independent_count(sol,edges,events,nonbinary,mapping):
 prednodes={(int(v['time']),int(v['label'])):k for k,v in sol.nodes(data=True)}
 assert len(prednodes)==len(sol) and all(sol.in_degree(k)<=1 for k in sol)
 inverse={prednodes[v]:k for k,v in mapping.items() if v in prednodes};pred_events={}
 for k in sol:
  cs=list(sol.successors(k))
  if len(cs)>=2:pred_events[k]=cs
 correct=0
 for k,cs in pred_events.items():
  if k in inverse and all(c in inverse for c in cs):
   a=inverse[k];b=frozenset(inverse[c] for c in cs)
   correct+=int(a in events and events[a]==b)
 ceiling=sum(a in mapping and all(b in mapping for b in bs) and mapping[a] in prednodes and all(mapping[b] in prednodes for b in bs) for a,bs in events.items())
 tpedge=sum(a in inverse and b in inverse and (inverse[a],inverse[b]) in edges for a,b in sol.edges)
 def f(tp,p,g):return 2*tp/(p+g) if p+g else None
 return dict(tp=correct,fp=len(pred_events)-correct,fn=len(events)-correct,division_f1=f(correct,len(pred_events),len(events)),gt_divisions=len(events),predicted_divisions=len(pred_events),matched_endpoint_ceiling=ceiling,excluded_nonbinary_gt_branches=len(nonbinary),edge_tp=tpedge,edge_fp=sol.number_of_edges()-tpedge,edge_fn=len(edges)-tpedge,edge_f1=f(tpedge,sol.number_of_edges(),len(edges)),gt_edges=len(edges),predicted_edges=sol.number_of_edges())

def graph_objective(g,sol,pairs,beta):
 total=0.
 for p in sol:
  cs=sorted(sol.successors(p));n=len(cs)
  if n:
   total+=sum(float(g[p][c]['weight']) for c in cs)+.75*(2-n)
   if n>=2:total+=beta*np.mean([pairs[p,a,b] for a,b in itertools.combinations(cs,2)])
 return float(total)

def audit_view(case,view):
 d=ROOT/'views'/case/view;fr=read(d/'FEATURES_DONE.json');assert sha(d/'features.npz')==fr['sha256']
 assert fr['input_admission_sha256']==sha(ROOT/'cases'/case/'ADMISSION.json')
 assert fr['masks_ready_sha256']==sha(ROOT/'cases'/case/'MASKS_READY.json')
 seal=read(ROOT/'SEALED_v2.json');assert fr['encoder_sha256']==sha(seal['encoder'])
 nn,ee,events,nonbinary,mp,audit,gaps=independent_mapping(case,fr['kept']);old=read(d/'GT_MAPPING.json')
 original=read(ROOT/'cases'/case/'ground_truth.json')['nodes']
 assert old['projected']['nodes']=={k:original[k] for k in nn}
 assert set(map(tuple,old['projected']['edges']))==ee
 assert {a:frozenset(bs) for a,bs in old['projected']['events']}==events
 assert old['mapping']=={k:list(v) for k,v in mp.items()}
 assert all(old['audit'].get(k,0)==audit.get(k,0) for k in set(old['audit'])|set(audit))
 assert old['projected']['annotation_gaps_or_absent_observed_ancestor']==gaps
 results={};model_hash={k:v['sha256'] for k,v in seal['models'].items()}
 for seed in range(1,6):
  gd=d/f's{seed}';done=read(gd/'EVALUATION_DONE.json');check(done['results']);gr=read(gd/'GRAPH_DONE.json');assert sha(gd/'graph.pkl')==gr['sha256']
  assert gr['features_receipt_sha256']==sha(d/'FEATURES_DONE.json')
  assert gr['model_sha256']==seal['bindings'][str(E/'runs'/f"Fluo-N2DL-HeLa_{'ChHead' if view.startswith('g1_') else 'R18'}_s{seed}"/'model.pt')]
  with (gd/'graph.pkl').open('rb') as f:g=pickle.load(f)
  with np.load(d/'features.npz') as z:assert {(int(v['time']),int(v['label'])) for k,v in g.nodes(data=True)}==set(zip(map(int,z['t']),map(int,z['seg_id'])))
  solutions={}
  for arm in ARMS:
   out=gd/arm;r=read(out/'RESULT.json');assert r['case']==case and r['seed']==seed and r['arm']==arm
   assert view==f"g{r['gap']}_o{r['offset']}" and fr['kept']==list(range(r['offset'],len(read(ROOT/'cases'/case/'ADMISSION.json')['images']),r['gap']))
   for p,k in [(out/'solution.pkl','solution_sha256'),(out/'SOLVER.json','solver_sha256'),(d/'GT_MAPPING.json','input_mapping_sha256'),(gd/'graph.pkl','graph_sha256'),(d/'features.npz','features_sha256')]:assert sha(p)==r[k]
   with (out/'solution.pkl').open('rb') as f:sol=pickle.load(f)
   assert set(sol)==set(g) and set(sol.edges)<=set(g.edges)
   assert all(sol.out_degree(p)<=3 and sol.in_degree(p)<=1 for p in sol)
   assert all(g.nodes[b]['time']==g.nodes[a]['time']+1 for a,b in sol.edges)
   assert independent_count(sol,ee,events,nonbinary,mp)==r['metrics'],(case,view,seed,arm)
   if arm in ['R','RG','ProgramPaintMatched','RenderedMatched']:
    assert r['model_sha256']==model_hash[f'{arm}_s{seed}']
    with np.load(out/'triplets.npz') as z:pair={(int(p),int(a),int(b)):float(v) for p,a,b,v in zip(z['mother'],z['daughter_a'],z['daughter_b'],z['residual'])}
    expected={(p,a,b) for p in g for a,b in itertools.combinations(sorted(g.successors(p)),2)};assert set(pair)==expected
    assert all(math.isfinite(v) and -4<=v<=4 for v in pair.values())
    beta=.5 if r['gap']==1 else .25;sr=read(out/'SOLVER.json');assert sr['alpha']==0 and sr['beta']==beta
    assert abs(graph_objective(g,sol,pair,beta)-sr['objective'])<1e-7
    assert abs(sum(f['objective'] for f in sr['frames'])-sr['objective'])<1e-7
    ref=solutions['TrackastraGraph'];assert all(sol.out_degree(p)>0 for p in ref if ref.out_degree(p)>0)
   solutions[arm]=sol;results[str(out/'RESULT.json')]=sha(out/'RESULT.json')
 return dict(case=case,view=view,status='PASS',time=time.time(),results=results,gt_binary_events=len(events),gt_nonbinary_branches=len(nonbinary),assignment=audit,mapping_sha256=sha(d/'GT_MAPPING.json'))

def seal_analysis():
 put(AUD/'ANALYSIS_LOCK.json',dict(prior_analysis_lock_sha256=sha(ROOT/'independent_v1/ANALYSIS_LOCK.json'),repair_receipt_sha256=sha(ROOT/'REPAIR_HOLD.json'),time=time.time(),source_sha256=sha(__file__),study_seal_sha256=sha(ROOT/'SEALED_v2.json'),
  primary_contrasts=['RG-R','RenderedMatched-ProgramPaintMatched'],rates=[1,8],
  complete_case_rule='Each rate uses all eligible named fields with all five seeds, all scheduled offsets and all six arms independently verified. Missing/held/failed fields listed. No score-based exclusions. Native/sparse common complete subset also reported.',
  field_macro='For fields with >=1 GT binary event, average defined F1 values across all eight offsets within seed, then all five seeds. Empty GT+prediction view gives undefined F1 and its count is reported; zero-GT fields included in pooled counts.',
  uncertainty='Paired field contrasts average offsets/seeds first. Percentile bootstrap 20000 within cell-line strata, fixed seed 20261101; report unadjusted95 and Bonferroni98.75% intervals for four contrasts. Descriptive field-level intervals, not independent biological-experiment evidence.',
  no_p_values=True,pre_aggregate_score_read=True,original_01_metrics='This independent dataset uses centroid-in-mask F1, not CTC mask-IoU BC(0).'))

def summarize():
 seal=read(ROOT/'SEALED_v2.json');rows=[];coverage=[];valid={1:[],8:[]}
 for rec in seal['cohort']:
  c=rec['case'];a=ROOT/'cases'/c/'ADMISSION.json';entry=dict(case=c,admitted=a.exists(),held=read(a)['held'] if a.exists() else None,complete={})
  for gap in [1,8]:
   done=all((AUD/'views'/c/f'g{gap}_o{o}.json').exists() for o in range(gap));entry['complete'][str(gap)]=done
   if done:
    valid[gap].append(c)
    for o,seed,arm in itertools.product(range(gap),range(1,6),ARMS):rows.append(read(ROOT/'views'/c/f'g{gap}_o{o}'/f's{seed}'/arm/'RESULT.json'))
  coverage.append(entry)
 fields={};contrasts={};pooled={}
 for gap in [1,8]:
  for case in valid[gap]:
   sub=[r for r in rows if r['case']==case and r['gap']==gap];groups={}
   for arm in ARMS:
    rr=[r for r in sub if r['arm']==arm];values=[r['metrics']['division_f1'] for r in rr];defined=[x for x in values if x is not None]
    seed_means=[]
    for seed in range(1,6):
     vv=[r['metrics']['division_f1'] for r in rr if r['seed']==seed and r['metrics']['division_f1'] is not None]
     if vv:seed_means.append(float(np.mean(vv)))
    groups[arm]=dict(mean=float(np.mean(seed_means)) if seed_means else None,seed_means=seed_means,undefined_views=len(values)-len(defined),gt_event_view_count=sum(x['metrics']['gt_divisions'] for x in rr)//5)
   fields[f'{case}/g{gap}']=groups
  for arm in ARMS:
   vals=[];counts=[]
   for seed,o in itertools.product(range(1,6),range(gap)):
    rr=[r for r in rows if (r['gap'],r['arm'],r['seed'],r['offset'])==(gap,arm,seed,o)];cc={k:sum(r['metrics'][k] for r in rr) for k in ['tp','fp','fn','edge_tp','edge_fp','edge_fn']};den=2*cc['tp']+cc['fp']+cc['fn'];cc.update(seed=seed,offset=o,f1=2*cc['tp']/den if den else None);counts.append(cc)
    if den:vals.append(cc['f1'])
   pooled[f'g{gap}/{arm}']=dict(mean=float(np.mean(vals)) if vals else None,by_seed_offset=counts)
  for aa,bb in [('RG','R'),('RenderedMatched','ProgramPaintMatched')]:
   cs=[c for c in valid[gap] if fields[f'{c}/g{gap}'][aa]['gt_event_view_count']>0];d=np.asarray([fields[f'{c}/g{gap}'][aa]['mean']-fields[f'{c}/g{gap}'][bb]['mean'] for c in cs],float)
   if not len(d):continue
   strata=defaultdict(list)
   for i,c in enumerate(cs):strata[c.split('-')[1]].append(i)
   rng=np.random.default_rng(20261101);boot=np.zeros(20000)
   for ix in strata.values():boot+=d[rng.choice(ix,size=(20000,len(ix)),replace=True)].sum(axis=1)
   boot/=len(d)
   contrasts[f'g{gap}/{aa}-{bb}']=dict(fields=cs,deltas=d.tolist(),mean=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ci98_75_bonferroni4=np.quantile(boot,[.00625,.99375]).tolist(),cell_line_counts={k:len(v) for k,v in strata.items()})
 put(AUD/'SUMMARY.json',dict(time=time.time(),coverage=coverage,complete_fields={str(k):len(v) for k,v in valid.items()},common_complete_fields=sorted(set(valid[1])&set(valid[8])),field_metrics=fields,pooled=pooled,contrasts=contrasts,analysis_lock_sha256=sha(AUD/'ANALYSIS_LOCK.json')))
 put(AUD/'VERIFIED.json',dict(time=time.time(),status='PASS_FOR_REPORTED_COMPLETE_FIELDS',summary_sha256=sha(AUD/'SUMMARY.json'),audit_receipts={str(p):sha(p) for p in AUD.glob('views/*/*.json')},study_queue_done_sha256=sha(ROOT/'QUEUE_DONE.json')))

def run():
 assert sha(__file__)==read(AUD/'ANALYSIS_LOCK.json')['source_sha256'];plan=read(ROOT/'SEALED_v2.json');check(plan['bindings']);check(plan['sources']);failed=set()
 while True:
  for c in plan['cohort']:
   case=c['case'];w=ROOT/'cases'/case;a=w/'ADMISSION.json'
   if not a.exists():continue
   if not read(a)['held'] and not (w/'MASKS_READY.json').exists():continue
   dest=AUD/'cases'/f'{case}.json'
   if not dest.exists():put(dest,audit_case(case))
   if read(a)['held']:continue
   for gap in [1,8]:
    for o in range(gap):
     view=f'g{gap}_o{o}';dest=AUD/'views'/case/f'{view}.json'
     if dest.exists():continue
     d=ROOT/'views'/case/view
     if not all((d/f's{s}'/'EVALUATION_DONE.json').exists() for s in range(1,6)):continue
     put(dest,audit_view(case,view))
  if (ROOT/'QUEUE_DONE.json').exists():break
  time.sleep(30)
 check(plan['bindings']);check(plan['sources']);summarize()
if __name__=='__main__':
 try:
  if sys.argv[1]=='seal':seal_analysis()
  elif sys.argv[1]=='run':run()
  else:raise ValueError(sys.argv[1])
 except BaseException:
  import traceback
  put(AUD/f'FAILURE_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
