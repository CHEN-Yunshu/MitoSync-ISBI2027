# Statistical interpretation and comparison map

The study is exploratory. CTC video 01 informed earlier development, and LFCT
comparisons and decoder follow-ups were added retrospectively. Sparse Pair
versus graph is the main endpoint emphasized in the current paper, not a newly
preregistered endpoint. Changing the narrative does not reset the testing
history or make the earlier test data unseen.

Five CTC seeds vary training initialization/sampling and the corresponding
Trackastra checkpoint, conditional on fixed corpora, encoders and videos.
Eight sparse offsets are averaged within each seed. Paired t intervals have
four degrees of freedom and describe training-repeat variation. They do not
measure generator-fit, sequence or biological-replicate uncertainty.

LFCT first averages defined offsets within seed, then five seeds within field,
then fields with equal weight. Paired bootstrap draws are stratified by cell
line. The 36 fields are not confirmed independent cultures or acquisition
batches. Intervals are conditional on this cohort and its resampling scheme.

The analysis families are preserved as originally reported. They are not one
study-wide error guarantee, and their existence does not establish prospective
specification. Released counts and contrast summaries retain null/negative comparisons.
Numerical recomputation remains available for the released record sets. We do not replace them with a favorable new
family or retroactively label any comparison confirmatory.

| Question | Existing comparison family | Interpretation |
|---|---|---|
| Images versus geometry, Full | R/RG x 2 datasets x 2 rates, eight-test Holm | Separate from the Pair-only family |
| Images versus geometry, Pair-only | R/RG x 2 datasets x 2 rates, eight-test Holm | Separate architecture, same question |
| Removing the parent head | R/RG x 2 datasets x 2 rates, eight-test Holm | Frozen pair scores |
| Direct painting versus rendering/M3 | 2 datasets x 2 rates, four-test Holm | Matched event and update budgets |
| Each matched synthetic branch versus real-only | 2 branches x 2 datasets x 2 rates, eight-test Holm | Does not establish either branch's advantage |
| Matched MitoSync versus SCF-adapted | 2 datasets x 2 rates, four-test Holm | Segmentation adaptation and generation cost differ |
| Additional M1 dynamics | PSC at 2 rates, two-test Holm | 40 paired programs |
| Production phase scheduling | 2 datasets x 2 rates, four-test Holm | Same downstream M3 and training budgets |
| LFCT R/RG versus graph | 2 arms x 2 rates, four-test Bonferroni intervals | Paired field bootstrap |
| LFCT synthesis contrasts | RG-R and rendered-painting x 2 rates, separate four-test family | Distinct training controls |
| LFCT fixed-score decoding | 3 contrasts x 2 rates, six-test Bonferroni intervals | Same frozen scores; no new model selection |

Per-family adjustments address only the comparisons in that family. Positive
results should be interpreted with the retrospective development history and
limited independent data. A future untouched-video study should fix one primary
endpoint and cohort before observing its scores, with the remaining analyses
marked secondary. Reversing the two previously used CTC sequences cannot create
an untouched test.

The public [CTC metric export](../results/ctc_association_metrics.json) preserves
120 historical summaries and all 600 associated seed values, including BC(0),
CT, TRA, DET, LNK and CHOTA. BC(1)/BC(2) must come from the temporal-tolerance
scorer; they cannot be reconstructed by changing the BC(0) threshold or rounding.
Those recorded scores and all 180 new continuity-off views are in
[CTC metrics](CTC_METRICS.md). The follow-up is descriptive and changes no model
or threshold; the original guarded configurations remain in the main table.

The paired seed intervals in the synthetic controls are unadjusted 95% t intervals;
Holm-adjusted p-values determine the family-wise significance decisions. Holm
adjustment does not turn these intervals into simultaneous intervals. The LFCT
Bonferroni bootstrap intervals use their explicitly stated wider quantiles.
`results/synthetic_controls.json` preserves the original means, seed differences,
95% intervals, p-values, family membership and source hashes without operational logs.
