# Data and models

| Material | Access | Included here |
|---|---|---|
| PSC and Fluo image sequences and annotations | [Cell Tracking Challenge](https://celltrackingchallenge.net/) | No; obtain the source data under the provider terms |
| LFCT data | Source described by Gachloo et al. in the paper; the evaluated fields and exclusions are listed in `reproducibility/field_dispositions.json` | No raw microscopy images |
| Generated MitoSync corpus, masks and lineage labels | Not currently downloadable from this repository | No full corpus yet |
| Trained LoRA, encoder, parent and pair checkpoints | Not currently downloadable from this repository | No checkpoints yet; SHA-256 identifiers are included |
| Training configurations and parent scaling parameters | `configs/` | Yes: 40 YAML files and 4 NPZ files |
| Per-view LFCT TP/FP/FN and comparison statistics | `reproducibility/` | Yes: all 14,580 records and all 53 field dispositions |
| CTC temporal-tolerance and association metrics | `results/` | Per-view counts, seed values and summaries |
| Author-review aggregate evidence | `reproducibility/identity/` and the paper | Aggregate results only; no reviewer notes, names or raw coordinates |

The current package supports implementation review, input/metric testing and
recalculation of reported statistical summaries. End-to-end prediction replay
also needs source images, masks, trained weights and full prediction/training
artifacts. Their absence is explicit; a checkpoint hash is not a downloadable
checkpoint.

The corpus release should include a file manifest, checksums, dataset-specific
source provenance, image/label pairing and a data license. No DOI, storage URL or
license is claimed before it exists.

The reference panels in the paper are real training images; the generated panels
are synthetic. Publishing example figures does not publish the entire dataset.
