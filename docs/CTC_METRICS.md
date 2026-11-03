# CTC tolerance and continuity controls

This follow-up uses the original CTC predictions and frozen scores. It adds no training,
threshold tuning or video selection. CTC development exposure remains unchanged.

## Protocol

The 450 historical views cover Graph, Pair R/RG and Full R/RG across two datasets,
five seeds, native rate and all eight sparse offsets. The 180 new views cover
Pair R/RG only and remove the continuity constraint while preserving candidates,
node sets, pair logits, beta and the utility. Alpha remains zero. Betas are
PSC .125/.25 and Fluo .5/.25 at native/sparse rates. Two CPU workers were used.

Each new solution is checked against an independent optimizer and export verifier.
Its unguarded objective must be at least the guarded objective. All result, solution
and solver receipts were re-read and bound to the final summary; 180 new views
and 450 historical views passed. This is not a new independent test.

## Division F1 with continuity removed

| Dataset/rate | Pair R, continuity | R, without | Difference (points) | Pair RG, continuity | RG, without | Difference (points) |
|---|---:|---:|---:|---:|---:|---:|
| PSC 1× | 0.874060 | 0.874060 | +0.0000 | 0.888138 | 0.888138 | +0.0000 |
| PSC 8× | 0.748845 | 0.749804 | +0.0960 | 0.754522 | 0.755539 | +0.1017 |
| FLUO 1× | 0.954325 | 0.943779 | -1.0546 | 0.950459 | 0.937222 | -1.3237 |
| FLUO 8× | 0.909020 | 0.908335 | -0.0685 | 0.913434 | 0.912923 | -0.0510 |

These are descriptive fixed-score differences, not a new significance family.
The original guarded configurations remain the main table; no configuration was
selected using these follow-up test results. The small sparse differences do not
attribute the full Pair gain to the continuity constraint. Native Fluo benefits
from continuity under these frozen settings.

## Temporal tolerance

BC(0), BC(1) and BC(2) use the same py-ctcmetrics event scorer and ground truth.
A tolerance of one means one **observed** frame: at gap 8 this spans eight original
frame intervals. BC(2) spans sixteen. Tolerance does not create new observations.

| Dataset/rate | Graph BC(0/1/2) | Pair R BC(0/1/2) |
|---|---|---|
| PSC 1× | 0.85655 / 0.85792 / 0.85827 | 0.87406 / 0.87477 / 0.87477 |
| PSC 8× | 0.69642 / 0.69904 / 0.69941 | 0.74884 / 0.75028 / 0.75088 |
| FLUO 1× | 0.94653 / 0.94653 / 0.94653 | 0.95432 / 0.95432 / 0.95432 |
| FLUO 8× | 0.82956 / 0.82986 / 0.82986 | 0.90902 / 0.90902 / 0.90902 |

The sparse Pair-vs-graph advantage persists at both nonzero tolerances. Each view
also contains TP/FP/FN, CT, TRA, DET and LNK; the historical seed export additionally
includes CHOTA. Offsets are averaged within seed before the five-seed summary.

`scripts/verify_ctc_metrics.py` recomputes these summaries from all 630 view records
and cross-checks 600 historical seed values. It does not rerun image prediction.
`reproducibility/ctc_followup_source.py` preserves the evaluation driver with only
its infrastructure root made configurable. Re-executing that orchestration requires
the archived preparation plans, candidate graphs, masks and checkpoints; these
large artifacts are not included in the compact release.

## Other historical controls

`results/synthetic_controls.json` retains six original control collections, including
104 paired contrasts. `scripts/verify_controls.py` checks their seed means, ordinary
95% t intervals, p-values and original Holm families. These 95% intervals must not
be labeled multiplicity-adjusted. No new testing family is substituted.
