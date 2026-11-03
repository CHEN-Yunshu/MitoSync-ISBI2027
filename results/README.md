# Paper tables and CTC metrics

Tables are extracted from the released LaTeX source and preserve display precision.
All data rows are included, including null and negative controls. Empty component
cells in Table 2 inherit the preceding component.

- Table 1: method; PSC native/sparse and Fluo native/sparse F1 and CT.
- Table 2: component and R/RG training; four CTC F1 conditions.
- Table 3: four CTC F1 conditions. The first three rows use Full with original sampling (Raw/RG/PM); the final two use Pair-only, matched-event direct painting and Render+M3.

`lfct_summary.csv` preserves the LFCT native/sparse summary and fixed-score solver/continuity comparisons, which are reported in the text of the current manuscript. It is an additional summary, not a numbered paper table.

R is real-only; RG is real plus generated training. Bold records displayed maxima,
not statistical significance. Rounded equality is not an equivalence test.

`ctc_association_metrics.json/csv` preserve 120 historical summaries and 600 seed
values for BC(0), CT, TRA, DET, LNK and CHOTA. `ctc_temporal_continuity.json/csv`
add BC(1)/BC(2), per-view TP/FP/FN, and the fixed-score continuity-off control.
[CTC metrics](../docs/CTC_METRICS.md) defines aggregation and the follow-up scope.
The [statistics map](../docs/STATISTICS.md) records comparison families and their
interpretation. LFCT unrounded per-view counts remain in `reproducibility/`.

`synthetic_controls.json` includes six historical control collections and 104
paired contrasts, preserving original ordinary 95% intervals and Holm p-values.
