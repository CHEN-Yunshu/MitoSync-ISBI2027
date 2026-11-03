# Paper tables

These files are transcribed programmatically from the released LaTeX source.
They preserve the reported precision and include every data row, including
controls that do not improve performance.

- Table 1: method; PSC native F1, CT; PSC sparse F1, CT; Fluo native F1, CT; Fluo sparse F1, CT.
- Table 2: component; data (R or RG); PSC native/sparse F1; Fluo native/sparse F1. Blank component cells inherit the preceding component.
- Table 3: data arm; PSC native/sparse F1; Fluo native/sparse F1. Raw/RG/PM use the full tracker and original sampling. Direct paint/Render+M3 use pair-only decoding with matched event budgets.

R means real-only training. RG means real plus generated training. Standard
deviations are over seeds. Rounded equality is not an equivalence test.
Bold formatting is not encoded as significance. See the paper for inference
families and the full unrounded LFCT records in `reproducibility/`.
