"""Recalculate released CTC summaries and fixed-score follow-up counts on CPU."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def main():
    old=read('results/ctc_association_metrics.json')['rows']
    assert len(old)==120
    for r in old:
        v=np.array(r['seed_values']);assert v.shape==(5,) and np.isfinite(v).all()
        np.testing.assert_allclose([r['mean'],r['sd']],[v.mean(),v.std(ddof=1)],rtol=0,atol=1e-12)
    report=read('results/ctc_temporal_continuity.json');rows=report['rows'];groups=report['groups']
    assert len(rows)==630 and len(groups)==28
    keys={(r['dataset'],r['gap'],r['offset'],r['seed'],r['arm']) for r in rows}
    assert len(keys)==len(rows)
    new=[r for r in rows if r['arm'].endswith('no continuity')]
    assert len(new)==180
    for r in rows:
        m=r['metrics']
        for j in range(3):
            tp,fp,fn=(m[f'{k}_div({j})'] for k in ['tp','fp','fn'])
            assert min(tp,fp,fn)>=0 and tp+fn==m['gt_divisions']
            den=2*tp+fp+fn;assert den>0
            assert abs(m[f'BC({j})']-2*tp/den)<1e-12
        assert m['BC(0)']<=m['BC(1)']+1e-12<=m['BC(2)']+2e-12
        if r in new:
            assert r['independent_objective']>=r['old_guarded_objective']-1e-6
    for key,metrics in groups.items():
        ds,g,arm=key.split('/');gap=int(g[1:]);rr=[r for r in rows if (r['dataset'],r['gap'],r['arm'])==(ds,gap,arm)]
        assert {(r['seed'],r['offset']) for r in rr}=={(s,o) for s in range(1,6) for o in range(gap)}
        assert len(rr)==5*gap
        for metric,d in metrics.items():
            v=np.array([np.mean([r['metrics'][metric] for r in rr if r['seed']==seed]) for seed in range(1,6)])
            np.testing.assert_allclose(d['values'],v,rtol=0,atol=1e-12)
            np.testing.assert_allclose([d['mean'],d['sd']],[v.mean(),v.std(ddof=1)],rtol=0,atol=1e-12)
    # Cross-bind the historical export and the independently recovered per-view record.
    for r in old:
        if r['metric']=='CHOTA':continue
        d=groups[f"{r['dataset']}/g{r['gap']}/{r['arm']}"][r['metric']]
        np.testing.assert_allclose(r['seed_values'],d['values'],rtol=0,atol=1e-12)
    assert report['receipts']['verification']['status']=='VERIFIED'
    assert report['receipts']['completion']['status']=='DONE'
    print('PASS: 120 historical summaries/600 seed values; 630 CTC views; BC(0/1/2) counts, aggregation and historical cross-binding; 180 frozen-score controls.')

if __name__=='__main__':main()
