"""Run inside the review ZIP: python reproducibility/verify_repro.py (NumPy)."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    raw = ROOT / 'lfct_original_counts.jsonl'
    expected = json.loads((HERE/'LFCT_ENDPOINT_DECOMPOSITION.json').read_text())
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == expected['source_sha256']
    pool = defaultdict(list)
    for line in raw.read_text().splitlines():
        row = json.loads(line); m = row['metrics']
        g, c, tp = m['gt_divisions'], m['matched_endpoint_ceiling'], m['tp']
        assert g-c+c-tp == m['fn'] and 0 <= tp <= c <= g
        values = [m['tp'], m['fp'], m['fn'], g, c, g-c, c-tp,
                  c/g if g else np.nan, tp/(tp+m['fp']) if tp+m['fp'] else np.nan,
                  m['division_f1'] if m['division_f1'] is not None else np.nan]
        pool[row['case'], row['gap'], row['arm'], row['seed']].append(values)
    names = ['tp','fp','fn','gt_divisions','matched_endpoint_ceiling',
             'unmatched_endpoint_events','matched_endpoint_missed_events',
             'endpoint_fraction','precision','division_f1']
    fields = defaultdict(list)
    for (case,gap,arm,seed), values in pool.items():
        assert len(values) == gap
        a = np.array(values, float)
        denom = np.isfinite(a).sum(axis=0)
        mean = np.divide(np.nansum(a,axis=0), denom, out=np.full(len(names),np.nan), where=denom>0)
        fields[case,gap,arm].append(mean)
    groups = defaultdict(list)
    for x in expected['per_field']:
        values = fields[x['case'],x['gap'],x['arm']]
        assert len(values) == 5
        mean = np.nanmean(values,axis=0)
        np.testing.assert_allclose(mean,[x[n] for n in names],rtol=0,atol=1e-12)
        groups[x['arm'],x['gap']].append(mean)
    for (arm,gap), values in groups.items():
        assert len(values) == 36
        x = expected['macro'][f'{arm}_g{gap}']
        np.testing.assert_allclose(np.nanmean(values,axis=0),[x[n] for n in names],rtol=0,atol=1e-12)
    geo = json.loads((HERE/'GEOMETRY_SOURCE.json').read_text())
    for group in geo['groups'].values():
        for metric in group.values():
            if isinstance(metric,dict) and 'full_minus_geometry' in metric:
                np.testing.assert_allclose(np.subtract(metric['full']['values'],metric['geometry']['values']),
                    metric['full_minus_geometry']['values'],rtol=0,atol=1e-12)
    for name, binding in json.loads((HERE/'SOURCE_BINDINGS.json').read_text()).items():
        assert hashlib.sha256((HERE/'source'/name).read_bytes()).hexdigest() == binding['packaged_sha256']
    print('PASS: FN decomposition, all per-field and macro statistics, geometry effect differences, source hashes')


if __name__ == '__main__':
    main()
