"""Run in the extracted review package: python recompute.py. Requires NumPy."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    expected=json.loads((ROOT/'decoder_summary.json').read_text());fields=expected['fields'];vals=defaultdict(lambda:defaultdict(list));seen=set()
    for name in ['lfct_original_counts.jsonl','lfct_decoder_counts.jsonl']:
        for line in (ROOT/name).read_text().splitlines():
            r=json.loads(line);key=(r['case'],r['gap'],r['offset'],r['seed'],r['arm']);assert key not in seen;seen.add(key)
            m=r['metrics'];den=2*m['tp']+m['fp']+m['fn'];f1=2*m['tp']/den if den else None;assert f1==m['division_f1']
            if f1 is not None:vals[(r['case'],r['gap'],r['arm'])][r['seed']].append(f1)
    assert len(seen)==14580
    means={key:float(np.mean([np.mean(seeds[s]) for s in range(1,6)])) for key,seeds in vals.items()}
    for key,arms in expected['field_metrics'].items():
        case,gap=key.rsplit('/g',1)
        for arm,value in arms.items():assert abs(means[case,int(gap),arm]-value)<1e-12
    strata=defaultdict(list)
    for i,c in enumerate(fields):strata[c.split('-')[1]].append(i)
    for key,c in expected['contrasts'].items():
        rate,comparison=key.split('/');a,b=comparison.split('-');gap=int(rate[1:]);d=np.array([means[f,gap,a]-means[f,gap,b] for f in fields]);rng=np.random.default_rng(20261103);boot=np.zeros(20000)
        for ix in strata.values():boot+=d[rng.choice(ix,(20000,len(ix)),replace=True)].sum(1)
        boot/=len(fields)
        np.testing.assert_allclose(np.quantile(boot,[.05/12,1-.05/12]),c['ci_bonferroni6'],rtol=0,atol=1e-12)
        assert abs(d.mean()-c['mean'])<1e-12
    identity=ROOT/'identity'
    if identity.exists():
        p=identity/'SUMMARY.json';d=json.loads(p.read_text());v=json.loads((identity/'VERIFIED.json').read_text());r=json.loads((identity/'READBACK_VERIFIED.json').read_text())
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==v['summary_sha256']==r['summary_sha256']
        assert d['groups']['all']['n']==8 and d['groups']['all']['by_label']['corrected']['exact_triplets']==r['counts']['corrected_exact']==7
        assert r['endpoint_conflicts']['corrected_same_daughter_id_at_both_clicks']==1
        assert d['selection_is_conditional'] and d['all_eight_in_denominator']
        print('PASS: fixed-endpoint aggregate/receipt consistency; raw-answer lookup is not included in this package')
    if (ROOT/'reproducibility/verify_repro.py').exists():
        subprocess.run([sys.executable,str(ROOT/'reproducibility/verify_repro.py')],check=True)
    print('PASS: 14,580 per-view count records, all field/seed/offset means, six adjusted decoder contrasts and package hashes')
if __name__=='__main__':main()
