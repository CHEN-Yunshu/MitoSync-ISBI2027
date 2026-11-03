# Methods and evaluation

This implementation snapshot specifies the input construction and LFCT evaluation behind
the paper. It adds no training runs, model selection or significance tests.
`reproducibility/reproducibility/source/` contains the relevant implementation. `SOURCE_BINDINGS.json` records
original and packaged SHA-256 hashes; only infrastructure paths are replaced in
the files marked `path_redacted`. Such orchestration files are inspectable source,
not standalone training commands. The portable feature/model/metric modules are
unchanged. The release manifest binds source and results to `paper/MitoSync.pdf`.

## 1. Complete daughter-pair input

For a candidate mother p and daughters a,b, A denotes mask area, x denotes the
rounded (y,x) mask centroid, r = sqrt(A/pi), d_a = x_a-x_p, d_b = x_b-x_p,
and m = (A_a d_a + A_b d_b)/(A_a+A_b). All positions and distances use pixels.
The ordered 25-dimensional geometry/context vector is:

| Index | Definition |
|---|---|
| 1 | log((A_a+A_b)/A_p) |
| 2 | abs(log(A_a/A_b)) |
| 3 | norm(m)/r_p |
| 4 | norm(x_a-x_b)/r_p |
| 5 | min(norm(d_a),norm(d_b))/r_p |
| 6 | max(norm(d_a),norm(d_b))/r_p |
| 7 | dot(d_a,d_b)/max(norm(d_a) norm(d_b),1e-8) |
| 8–13 | h(p), the six node values below |
| 14–19 | (h(a)+h(b))/2 |
| 20–25 | abs(h(a)-h(b)) |

In order, h(i) contains log(A_i); nearest other centroid distance divided by r_i
(zero for a singleton frame); number of other centroids within 4 r_i; a first
observed frame flag; a last observed frame flag; and log2(gap)/3. Neighbors are
computed within the same observed frame. Node context is cached in float16 and
cast to float32 for pair assembly. The assembled vector is float32 and clipped
elementwise to [-12,12]. There is no additional z-score normalization for this
pair vector. The optional parent head has its own preprocessing and should not
be conflated with this definition. See `triplets.py`, `features.py`, `targets.py`.

The three image embeddings have 256 entries each. A shared Linear(256,32)+GELU
produces m_v,a_v,b_v. The network concatenates m_v, (a_v+b_v)/2, abs(a_v-b_v),
a_v*b_v, abs(m_v-(a_v+b_v)/2), m_v*((a_v+b_v)/2), and the 25 context values.
The resulting 217 entries pass through Linear(217,128), GELU, Dropout(0.1),
Linear(128,64), GELU and Linear(64,1). Inference clips the scalar logit to [-4,4].

Every geometry entry and visual combination is invariant to exchanging a,b.
Thus the full pair score, not only a subset of its image features, is invariant
in evaluation mode. Dropout during training does not imply that independent
stochastic forward calls must be identical. Identity labels and ground-truth
parentage are used to create training targets, not to construct input features.
The included tests swap both daughter embeddings and their context inputs,
and check that changes to training IDs do not alter observed descriptors.

## 2. Observed frames, crops and frozen encoder

Temporal subsampling occurs before feature extraction. At observed index j,
the three channels are max(j-1,0), j, min(j+1,last). For sparse gap 8, an interior
cell observed at original frame t therefore uses original frames t-8,t,t+8,
not hidden frames t-1,t,t+1. At either endpoint the nearest observed frame is
repeated. The same center from the central mask is used in all three channels;
no ground-truth motion alignment is applied. Each candidate cell has its own
three-frame crop, including daughters centered at the next observed index.

Each observed image is normalized independently: map its 1st and 99.5th
percentiles to 0 and 255, clamp, then round with floor(value+0.5); use 1e-6 as
the minimum percentile range. Centroids use floor(mean coordinate+0.5).
Image crops are 48x48 pixels with reflection padding; the central instance mask
uses zero padding, avoiding reflected duplicate identities. The uint8 images
are divided by 255, normalized by ImageNet channel means (0.485,0.456,0.406)
and standard deviations (0.229,0.224,0.225), and resized to 96x96 by bilinear
interpolation with align_corners=False. See `encoder.py` and `features.py`.

The encoder comes from the final real-only, seed-1 division-classifier checkpoint
for the corresponding CTC dataset. It is fixed across all five pair-head seeds.
The LFCT experiment uses the Fluo encoder. Checkpoint hashes are in
`TRAINING_AND_MODEL_BINDINGS.json`. Classifier pretraining starts from torchvision
ResNet-18 IMAGENET1K_V1 with a Linear(512,1) output, using real training-split
crops (PSC video 02 frames 0–254; Fluo video 02 frames 0–77, with complete
three-frame contexts within that split). Training uses 6,000 updates, batch 256,
AdamW lr 3e-4, weight decay 1e-4, betas (0.9,0.999), cosine decay without warmup,
binary cross-entropy with logits, and bf16 autocast with fp32 parameters.
Batch normalization is trained. The expected positive:negative sampling ratio
is 1:10, sampled with replacement; the fractional positive count is randomized.
All three channels share horizontal/vertical flips (each p=0.5) and a uniformly
sampled multiple of 90 degrees. There is no early stopping or checkpoint choice.
The positive target is the mother at its last frame before two annotated
daughters appear in the next frame. Segmentation instances obtain training
identities by maximum mask overlap with one-to-one ownership; unmatched
instances are negative classifier candidates. This target construction is
separate from the observed-only feature path. See `encoder_training.py`,
`encoder_common.py` and `encoder_candidates.py`.

After pretraining, the encoder is frozen in evaluation mode and truncated after
layer2 (128 channels). The central binary instance mask is resized to that map
by area interpolation. Weighted averages inside the mask and its complement
are separately L2-normalized (eps=1e-8), then concatenated to 256 entries.
The classifier itself was trained on image crops; mask-conditioned pooling is
introduced when extracting the frozen features. Cached embeddings are float16.

Pair-head training uses 6,000 updates, batch 64, AdamW lr 0.001, weight decay
0.01, cosine decay, gradient-norm clipping at 5, and balanced binary
cross-entropy. Each batch equally samples gaps 1,2,4,8 and positive/negative
targets. R draws 64 real examples; RG draws 32 real and 32 synthetic examples.
Seeds 1–5 vary initialization and sampling. Training candidates use up to ten
previous-frame parents within 256 pixels; negative daughter pairs come from the
four nearest children, and all available true training pairs are included.
At inference, pairs are enumerated from the Trackastra candidate graph without
ground-truth insertion. The exact sample-plan digests and original checkpoint
hashes are included. The decoder protocol specifies all LFCT weights and controls.

## 3. LFCT metric and exclusions

The metric is exact binary mother–daughter event F1 under centroid-in-mask
assignment. It is not CTC BC(0), and its absolute values should not be compared
directly with the CTC scores. All compared methods share images, masks, cohort
and mapping rules. `lfct_metrics.py` implements the following rules.

Observed frames are selected first. Each retained annotation node is linked to
its unique ancestor at the immediately preceding observed frame, when one
exists. Missing annotation paths remain disconnected. Multiple-parent source
annotations are rejected rather than selecting a convenient parent. A projected
ground-truth division has exactly two children; projected branches with more
children are counted separately and excluded from the binary-event denominator.

A reference centroid can match only the detected mask containing its rounded
pixel (floor(coordinate+0.5)). Both the floating coordinate and rounded pixel
must lie inside the image. Background, outside-image points and unassigned
points remain unmatched; there is no nearest-cell substitution or clipping to
the image boundary. If several reference points fall in one mask, the point
nearest its centroid wins, with the smaller integer point ID breaking a tie.
The others remain unmatched, giving one-to-one assignment.

A predicted branch is a node with at least two children. It is a true positive
only if its mother and complete unordered child set exactly match a mapped
binary ground-truth event. A three-child prediction cannot match a two-child
event. Unmatched ground-truth endpoints do not remove the event: it remains a
false negative. FP is the number of predicted branches minus TP; FN is the
number of binary reference events minus TP; F1=2TP/(2TP+FP+FN).

For a view with no reference division, false predicted divisions give F1=0;
if there is also no predicted division, F1 is undefined (None), not one.
Defined offsets are averaged within each seed, then five seeds within each
field, then fields with equal weight. The contrast cohort uses fields with
reference divisions; pooled TP/FP/FN also retain zero-division views. In the
actual 36-field cohort every evaluated native/sparse view has a reference event
and a defined F1. Five seeds and eight sparse offsets are correlated views,
not additional independent specimens.

All 53 requested fields are listed in `field_dispositions.json` and
`LFCT_HIERARCHY.json`: 36 accepted, 11 with multiple-parent annotations, five
integrity failures, and one unavailable source. The five integrity failures are:

| Field | Input failure |
|---|---|
| LFCT-hek293-10x-FLD_3 | Archive frame count differs from XML ImageData.nframes |
| LFCT-hek293-10x-FLD_4 | Archive frame count differs from XML ImageData.nframes |
| LFCT-hek293-10x-FLD_5 | Archive frame count differs from XML ImageData.nframes |
| LFCT-hek293-10x-FLD_6 | Archive frame count differs from XML ImageData.nframes |
| LFCT-hek293-20x-FLD_8 | RFP frames are spatially constant; no usable target-channel signal |

The four frame-count mismatches were not truncated, shifted or relabeled because
their alignment was unverified. No alternate channel was selected for FLD_8.
LFCT-u87-20x-FLD_1 was unavailable. These input dispositions are shared across
methods and precede the comparisons, rather than depending on tracking scores.

Bootstrap resampling is paired by field and stratified by cell line (20,000
draws). The graph comparisons use four-test Bonferroni intervals; the separate
fixed-score decoder family has six tests, as specified in `decoder_protocol.json`.
The reported intervals describe uncertainty under resampling of this field
cohort. Well, culture and acquisition-batch mappings are unavailable in the
local evidence, so field independence and biological replication are not
established. Cell-line direction consistency does not resolve that limitation.

## 4. What limits the absolute LFCT score?

For each view let G be binary ground-truth events, C the events with all three
endpoints mapped to solution nodes, and TP the correctly linked events. Then
FN=(G-C)+(C-TP). The first term reflects missing or ambiguous endpoint mapping
(including detection and point-to-mask assignment); the second reflects missed
events despite available endpoints, including candidate and association errors.
This is a count decomposition, not causal attribution to a particular module.
C does not guarantee that both required candidate edges exist. It is also not
an estimate of segmentation accuracy.

For the frozen real-only pair model R, equal-field mean counts are:

| Quantity | Native | Sparse |
|---|---:|---:|
| Ground-truth events G | 33.89 | 30.77 |
| All endpoints mapped C | 16.33 | 16.06 |
| Correctly linked TP | 9.56 | 4.58 |
| Unmatched-endpoint events G-C | 17.56 | 14.72 |
| Missed despite mapped endpoints C-TP | 6.78 | 11.48 |
| False predicted events FP | 155.27 | 43.98 |
| Event F1 | .130 | .138 |

Counts are averaged over offsets, seeds and fields; they are not unique pooled
events. The mean per-view endpoint coverage is 54.2%/57.5%, and precision is
11.2%/15.0%. Thus endpoint matching, missed associations and false branches all
limit the score. Sparse improvement over graph does not demonstrate reliable
end-to-end lineage recovery. All arms and per-field quantities are in
`LFCT_ENDPOINT_DECOMPOSITION.json`; `verify_repro.py` independently recalculates
them from the included records. No new model or post-hoc threshold was selected.

## 5. Configuration-specific image-feature effects

The full-minus-geometry changes below are percentage points from the unrounded
five-seed summaries. Holm correction covers the eight comparisons together.

| Configuration | PSC native | PSC sparse | Fluo native | Fluo sparse |
|---|---:|---:|---:|---:|
| R | 1.571 | 1.708 | 0.482 | 1.251 |
| RG | 0.635 | 1.204 | 0.000 | 1.078 |

The manuscript range 0.64–1.20 applies to RG on PSC and sparse Fluo only; the
corresponding R range is 1.25–1.71. Both native Fluo comparisons are nonsignificant
in that family. In the separate eight-test no-parent family, RG gains at sparse
rate are 3.16179/1.83147 points on PSC/Fluo. The full original summary, including
all seeds and null results, is supplied as `GEOMETRY_SOURCE.json`.

The decoder evidence is kept separate. With frozen R scores and continuity
disabled, the comparison is:

| ILP minus greedy | Gain (F1 points) | Six-test adjusted interval |
|---|---:|---:|
| Native | +0.45 | [0.21,0.75] |
| Sparse | +1.45 | [0.82,2.17] |

This compares solvers
under the same utility; it is not a reproduction or comparison against the
complete method of Zyss et al. Enabling continuity changes sparse R/RG by
-0.25/-0.29 points and does not explain the positive sparse benefit.


## Configuration and artifact locations

The 20 original Trackastra architecture/training pairs and four fitted parent
normalization arrays are in `configs/`. Native parent geometry has 9 edge
values and 3 node values; sparse geometry has 15 edge and 6 node values.
These arrays are separate from the 25-dimensional pair descriptor.

The table definitions above refer to files beneath `reproducibility/`.
The raw per-view counts are preserved, including all comparison arms.
Images, model checkpoints and complete generation corpora are not bundled.
See [Data and models](DATA.md) and [Reproduction](REPRODUCING.md).
