# Code guide

## Association and evaluation

The implementation used for the inspected results is preserved with a source
manifest. Start with these files:

| Component | File |
|---|---|
| Symmetric daughter-pair network | `reproducibility/reproducibility/source/model.py` |
| Complete 25-value descriptor | `reproducibility/reproducibility/source/triplets.py` |
| Observed-frame crops and embeddings | `reproducibility/reproducibility/source/features.py`, `encoder.py` |
| Pair-head training | `reproducibility/reproducibility/source/pair_training.py` |
| Frozen-score inference | `reproducibility/audit/source/inference.py` |
| Joint child-set ILP decoder | `reproducibility/method/decoder.py` |
| Native/sparse parent source | `reproducibility/audit/source/image_parent_20261024/`, `multirate_parent_20261025/` |
| LFCT metric and projection | `reproducibility/reproducibility/source/lfct_metrics.py`, `lineage_projection.py` |

The date-bearing parent directories identify the evaluated implementations; they
are retained to avoid silently substituting a different model. Source identifiers
and original/current hashes are recorded in `SOURCE_PROVENANCE.json`. Some
training/evaluation orchestration still refers to the documented study layout;
it is inspectable research source, not a packaged one-command benchmark runner.

## Programming, rendering and label alignment

- `code/mitosync/kin_fit.py`, `kinstats.py`: fit training-split dynamics.
- `code/mitosync/lineage_sim_v3.py`: program cell identities and division times.
- `code/vace/cond_render.py`: render conditioning masks.
- `code/vace/train_lora.py`, `train.py`: VACE LoRA training.
- `code/vace/gen_corpus.py`: generate videos with a trained adapter.
- `code/vace/relabel_by_readout.py`: keep, re-time or remove division labels.
- `code/mitosync/certify.py`: retry policy and clip-level acceptance.
- `code/mitosync/paint_render.py`: direct rendering of programmed masks.

These scripts use a configurable workspace instead of the authors' server paths:

```bash
python scripts/setup_workspace.py --root /path/to/mitosync-workspace
export MITOSYNC_ROOT=/path/to/mitosync-workspace
export N="$MITOSYNC_ROOT"
export CELLPOSE_PYTHON=/path/to/cellpose-environment/bin/python
export VACE_PYTHON=/path/to/vace-environment/bin/python
```

The setup command only creates workspace directories and links to the bundled
source. Supply the CTC inputs, fitted trajectory library, intensity mapping and
external backbone/checkpoint files before generation. The recorded package lists
are in `environments/`; the GPU generation and tracking environments are separate.
Use each script's command-line help for its required inputs and output paths.

The historical complete experiment history is not a dependency of public
verification. Alternative exploratory launchers, monitoring processes and
cleanup scripts are not part of this release.
