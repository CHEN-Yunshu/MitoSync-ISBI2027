"""Recalculate released historical seed contrasts without redefining test families."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
def main():
    families=json.loads((ROOT/'results/synthetic_controls.json').read_text())['families'];checked=[]
    def walk(x):
        if isinstance(x,dict):
            if all(k in x for k in ['values','mean','ci95','p_unadjusted']):
                a=np.array(x['values']);se=a.std(ddof=1)/np.sqrt(len(a))
                np.testing.assert_allclose(x['mean'],a.mean(),rtol=0,atol=1e-12)
                np.testing.assert_allclose(x['ci95'],a.mean()+np.array([-1,1])*t.ppf(.975,len(a)-1)*se,rtol=0,atol=1e-10)
                if se>0:np.testing.assert_allclose(x['p_unadjusted'],2*t.sf(abs(a.mean()/se),len(a)-1),rtol=0,atol=1e-10)
                checked.append(x)
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for f in families.values():walk(f)
    def holm(rows,key='p_holm'):
        order=sorted(range(len(rows)),key=lambda i:rows[i]['p_unadjusted']);prior=0.
        for rank,i in enumerate(order):
            prior=max(prior,min(1.,(len(rows)-rank)*rows[i]['p_unadjusted']))
            np.testing.assert_allclose(rows[i][key],prior,rtol=0,atol=1e-10)
    for name in ['M1_dynamics','M2_schedule','SCF_adapted']:holm(list(families[name]['contrasts'].values()))
    holm(list(families['MAGIK_balanced']['contrasts'].values()),'p_holm2')
    direct=families['Direct_paint']
    for group in ['primary_family','vs_real_family']:holm([direct['contrasts'][k] for k in direct[group]])
    groups=families['Parent_removal']['groups']
    for metric in next(iter(groups.values())):
        rr=[g[metric]['full_minus_no_parent'] for g in groups.values()]
        if all('p_holm' in r for r in rr):holm(rr)
    assert len(checked)==104
    print('PASS: 104 historical paired contrasts, 95% t intervals and original Holm families; no new significance tests.')
if __name__=='__main__':main()
