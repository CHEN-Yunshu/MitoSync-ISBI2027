"""Independent (verifier-written) CTC lineage/layout check used to audit recombine.py / make_slice outputs. Edit the check(...) calls at the bottom; run with the cellres env python."""
import sys, json, re
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage as ndi
D = Path('STUDY_ROOT/data/ctc')
def tracks(p):
    t = np.loadtxt(p, dtype=int).reshape(-1,4); return {int(L):(int(B),int(E),int(P)) for L,B,E,P in t}
def check(name, src_ds=None, src_offset=0, min_copy_id=None):
    vd = D/name/'02'; gt = D/name/'02_GT/TRA'; st = D/name/'02_ST/SEG'
    tr = tracks(gt/'man_track.txt'); T = len(list(vd.glob('t*.tif')))
    assert T == len(list(st.glob('man_seg*.tif'))) == len(list(gt.glob('man_track*.tif'))), 'frame count mismatch'
    # division events: parents with >=1 child; count children per parent
    from collections import Counter
    pc = Counter(P for L,(B,E,P) in tr.items() if P)
    bad_par = [(L,P) for L,(B,E,P) in tr.items() if P and (P not in tr or tr[P][1] != B-1)]
    res = dict(name=name, frames=T, tracks=len(tr), max_id=max(tr), daughters=sum(pc.values()),
               parents_with_children=len(pc), parents_2ch=sum(1 for v in pc.values() if v==2), parents_1ch=sum(1 for v in pc.values() if v==1),
               parents_3plus=sum(1 for v in pc.values() if v>2), bad_parent_links=len(bad_par), bad_parent_examples=bad_par[:5])
    if min_copy_id: 
        pcc = Counter(P for L,(B,E,P) in tr.items() if P and L>=min_copy_id)
        res.update(copy_tracks=sum(1 for L in tr if L>=min_copy_id), copy_daughters=sum(pcc.values()), copy_parents=len(pcc), copy_parents_2ch=sum(1 for v in pcc.values() if v==2))
    # per-frame: ST-only presence (no TRA help) for copy ids; multi-component for copy ids; ids outside [B,E]; marker-mask conflicts
    st_missing=0; st_extra=0; multi_copy=0; marker_conflict=0; dtype=None; shape=None; img_changed_outside=0; max_label=0; frames_checked=0
    sample = range(0,T,max(1,T//25))
    for t in sample:
        seg = tifffile.imread(st/f'man_seg{t:03d}.tif'); tra = tifffile.imread(gt/f'man_track{t:03d}.tif'); img = tifffile.imread(vd/f't{t:03d}.tif')
        dtype = (str(img.dtype), str(seg.dtype), str(tra.dtype)); shape=img.shape; max_label=max(max_label,int(seg.max()),int(tra.max()))
        exp = {L for L,(B,E,P) in tr.items() if B<=t<=E}
        ids_st = set(int(i) for i in np.unique(seg) if i); ids_tra = set(int(i) for i in np.unique(tra) if i)
        # every marker id and seg id must be an expected id
        st_extra += len((ids_st|ids_tra)-exp)
        # expected ids missing from BOTH seg and tra
        st_missing += len(exp-(ids_st|ids_tra))
        # marker pixels lying on a seg mask with a different nonzero id
        m = (tra>0)&(seg>0)&(tra!=seg); marker_conflict += int(m.sum())
        if min_copy_id:
            for i in ids_st:
                if i>=min_copy_id and ndi.label(seg==i)[1]>1: multi_copy+=1
            # copy ids must be present in ST (not only via marker)
            st_missing_copy = len({i for i in exp if i>=min_copy_id} - ids_st)
            res['copy_missing_in_ST'] = res.get('copy_missing_in_ST',0)+st_missing_copy
        if src_ds:
            simg = tifffile.imread(D/src_ds/'02'/f't{t+src_offset:03d}.tif'); sseg = tifffile.imread(D/src_ds/'02_ST/SEG'/f'man_seg{t+src_offset:03d}.tif')
            assert simg.shape==img.shape and simg.dtype==img.dtype, ('src shape/dtype differ', simg.shape, img.shape, simg.dtype, img.dtype)
            if min_copy_id:
                pasted = seg>=min_copy_id; 
                # original ST labels must be untouched where no copy pasted
                res['orig_seg_changed'] = res.get('orig_seg_changed',0)+int(((sseg!=seg)&~pasted).sum())
                res['copy_over_orig_seg'] = res.get('copy_over_orig_seg',0)+int((pasted&(sseg>0)).sum())
                far = ~ndi.binary_dilation(pasted, iterations=4)
                img_changed_outside += int((simg!=img)[far].sum())
                res['pasted_px_frac'] = res.get('pasted_px_frac',0)+pasted.mean()/len(sample)
            else:
                img_changed_outside += int((simg!=img).sum()); res['seg_identical_to_src'] = res.get('seg_identical_to_src',True) and bool((sseg==seg).all())
        frames_checked+=1
    res.update(frames_checked=frames_checked, dtypes=dtype, shape=shape, max_label=max_label, label_fits_uint16=max_label<65536,
               ids_outside_BE=st_extra, ids_missing_in_both=st_missing, marker_on_other_mask_px=marker_conflict, multi_component_copy_ids=multi_copy, img_px_changed_outside_paste=img_changed_outside)
    print(json.dumps(res))






check('PhC-C2DL-PSC_recomb', src_ds='PhC-C2DL-PSC', min_copy_id=17609)
check('PhC-C2DL-PSC_recomb_div', src_ds='PhC-C2DL-PSC', min_copy_id=17609)
check('Fluo-N2DL-HeLa_recomb', src_ds='Fluo-N2DL-HeLa', min_copy_id=1947)
check('Fluo-N2DL-HeLa_recomb_div', src_ds='Fluo-N2DL-HeLa', min_copy_id=1947)
