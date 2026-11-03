# Recorded configurations and parent normalization

`runs/` contains architecture and training YAMLs for the 20 Trackastra graph
checkpoints: PSC/Fluo, ChHead/R18 and seeds 1–5. The other four directories hold
the fitted native/sparse parent normalization arrays. These are small numeric
parameters, not image datasets or learned network weights.

All original file hashes matched historical preparation receipts. Architecture
YAMLs and NPZs are byte-identical. Training YAMLs replace only the machine-specific
root with STUDY_ROOT; replace that token with the chosen workspace before use.
`BINDINGS.json` distinguishes original and portable hashes.
`CONFIGURATION_SUMMARY.json` gives every parsed setting, scaling value and bound
checkpoint identifier.

These graph settings are distinct from the 6,000-update parent/pair-head and
encoder settings in `docs/METHODS.md`. Native parent arrays contain 9 edge and 3
node values; sparse contains 15 and 6. The pair descriptor has 25 values with
no use of these parent z-score arrays.
