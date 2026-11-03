# Reproducing the released evidence

Run `python scripts/verify_release.py` after installing `requirements-audit.txt`.
The verifier checks the file manifest, model/configuration bindings, four fitted
normalization layouts, complete daughter exchange invariance, observed sparse
frame indexing, candidate pruning, decoder consistency and LFCT scoring.
It recalculates all 14,580 records and the specified confidence intervals.
Twenty synthetic contract tests use no training images or model weights.

The 36 retained LFCT fields, 17 exclusions and their reasons are in
`reproducibility/field_dispositions.json`. No reference division is removed
because its mask is missing. Undefined zero-event F1 and deterministic ambiguous
assignment handling are defined in [Methods](METHODS.md).

`results/paper_tables.json` preserves the display precision and configuration
labels of Tables 1–3; these rounded tables must not be used to reconstruct paired
confidence intervals. `reproducibility/` contains the unrounded records for its
reported analyses, including null and negative contrasts.

The graph configurations in `configs/runs/` correspond to native ChHead and sparse
R18 backbones, five seeds for each dataset. Their original training files matched
the historical checkpoint receipts before path sanitization. Replace STUDY_ROOT
with the chosen workspace before running those YAMLs. Native parent geometry
uses 9 edge/3 node values and sparse uses 15 edge/6 node values; the pair vector
is 25-dimensional and does not use these z-score arrays.

Recorded `resume: true` and `git_commit: none` values are preserved. Saved
configuration files do not reconstruct historical optimizer/RNG/cache states.
The input lists contain video-02 training/validation data, but do not negate
the disclosed historical development use of video 01.

## What this release does not claim

The checks do not regenerate images, refit models or rerun full prediction.
Those operations require the separately listed data and weights. Field-level
intervals are not confirmed biological-replicate intervals. Seven matches in
eight selected identity checks are a targeted diagnostic, not corpus accuracy.
