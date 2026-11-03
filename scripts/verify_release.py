"""Run from the repository: python scripts/verify_release.py."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for name,h in read(ROOT/'RELEASE_MANIFEST.json').items():assert sha(ROOT/name)==h,name
    config=ROOT/'configs';bindings=read(config/'BINDINGS.json');summary=read(config/'CONFIGURATION_SUMMARY.json')
    assert len(bindings['files'])==44 and len(summary['runs'])==20
    assert sum(v['bytes_read'] for v in bindings['files'].values())==41756
    for name,r in bindings['files'].items():
        p=config/name
        assert sha(p)==r['packaged_sha256'] and r['historical_sha256']==r['original_sha256']
        assert r['root_prefix_replaced']==(r['kind']=='tracker_training')
        if not r['root_prefix_replaced']:assert sha(p)==r['original_sha256']
    model_bindings=read(ROOT/'reproducibility/reproducibility/TRAINING_AND_MODEL_BINDINGS.json')['graph_model_and_config_sha256']
    common=[]
    for name,r in summary['runs'].items():
        model=yaml.safe_load((config/'runs'/name/'config.yaml').read_text())
        train=yaml.safe_load((config/'runs'/name/'train_config.yaml').read_text())
        assert model==r['config.yaml'] and train==r['train_config.yaml']
        assert train['seed']==int(name.rsplit('_s',1)[1]) and train['name']==name
        assert model['coord_dim']==train['ndim']==2 and model['feat_dim']==7 and model['nhead']==4
        for k in set(model)&set(train):assert model[k]==train[k],(name,k)
        assert model_bindings[f'runs/{name}/model.pt']==r['checkpoint_sha256']
        assert model_bindings[f'runs/{name}/train_config.yaml']==bindings['files'][f'runs/{name}/train_config.yaml']['original_sha256']
        c={k:v for k,v in train.items() if k not in ['name','seed','input_train','input_val','cachedir']}
        assert c==summary['common_tracker_training'];common.append(c)
        ds,arm,seed=name.rsplit('_',2)
        inputs=[f'STUDY_ROOT/data/ctc/{ds}_trainhead/02']
        inputs+=([f'STUDY_ROOT/data/ctc_head/{ds}_recomb/02'] if arm=='ChHead' else [f'STUDY_ROOT/data/ctc_head/{ds}_d8o{i}/02' for i in range(8)])
        assert train['input_train']==inputs
        assert train['input_val']==[f'STUDY_ROOT/data/ctc/{ds}_valtail/02']
    assert set(summary['parent_scaling'])=={'psc_native','fluo_native','psc_sparse','fluo_sparse'}
    for name,r in summary['parent_scaling'].items():
        with np.load(config/r['file'],allow_pickle=False) as z:
            assert set(z.files)=={'edge_mean','edge_sd','node_mean','node_sd'}
            for k in z.files:
                expected=(15 if k.startswith('edge') else 6) if name.endswith('_sparse') else (9 if k.startswith('edge') else 3)
                assert z[k].shape==(expected,) and z[k].dtype==np.float32 and np.isfinite(z[k]).all()
                np.testing.assert_array_equal(z[k],np.array(r['values'][k],np.float32))
                if k.endswith('_sd'):assert (z[k]>0).all()
            if name.endswith('_sparse'):
                for kind,idx in [('edge',[3,4,5,9,10,11]),('node',[3,4,5])]:
                    np.testing.assert_array_equal(z[kind+'_mean'][idx],np.zeros(len(idx)))
                    np.testing.assert_array_equal(z[kind+'_sd'][idx],np.ones(len(idx)))
    print('PASS: 44 configuration/parameter artifacts; 20 checkpoint/config pairs; four normalization layouts and values',flush=True)
    r=ROOT/'reproducibility'
    for script in ['recompute.py','reproducibility/test_input_contract.py','audit/test_full_inference.py']:
        subprocess.run([sys.executable,str(r/script)],cwd=r,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_ctc_metrics.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_controls.py')],check=True)
    print('PASS: public release hashes, 44 configurations, 14,580 LFCT records, 630 CTC views, 600 historical seed values and 20 contract tests; no end-to-end prediction replay.')

if __name__=='__main__':main()
