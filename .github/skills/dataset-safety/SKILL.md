---
name: dataset-safety
description: Reusable Fakeddit safeguards for immutable source data, traceable processing, official split preservation, and leakage prevention.
---

# Dataset Safety Procedure

1. Treat `data/raw/` as read-only. Before any write-capable command, inspect its target path and reject raw-data edits, deletion, replacement, or in-place conversion.
2. Treat `data/processed/train_clean.csv`, `validation_clean.csv`, and `test_clean.csv` as source split artifacts. Do not rewrite them during modeling; derived subsets and manifests belong in a separate experiment directory.
3. Load each official split independently. Record source CSV, row count, label distribution, and deterministic selection seed/IDs for any subset.
4. Verify ID non-nullness and uniqueness within each split, then compute pairwise ID intersections. Report the exact scope: zero matching IDs is not a proof against near-duplicate text/images or other forms of contamination.
5. Join image files to rows by exact sample ID and expected naming convention. Verify files are readable; exclude unavailable/corrupt images without copying another sample into their place.
6. Audit preprocessing for global fitting. Fit learned text/vector transforms on train only; apply the fitted transform to validation/test. Do not use held-out split statistics to fit preprocessing.
7. Do not use test labels, predictions, metrics, or examples in training, validation, checkpoint selection, or tuning. Load test only for final evaluation after all choices are fixed.
8. Keep split IDs/manifests and derived outputs separate. Do not write new data over existing experiment results without explicit authorization.
9. If provenance or split membership cannot be confirmed, stop the experiment or clearly mark it unevaluated; do not silently infer membership.
