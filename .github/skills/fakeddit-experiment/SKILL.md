---
name: fakeddit-experiment
description: Reusable procedure for preparing, training, selecting, and evaluating a Fakeddit experiment without split leakage; use when implementing or running a model experiment.
---

# Fakeddit Experiment Procedure

1. **Inventory:** Inspect the experiment script, exact CSV inputs, schema, requirements, model/result output paths, and existing artifacts. Refuse to overwrite result files without explicit user authorization.
2. **Validate data:** Read only needed columns first. Check `id`, `text`/image path, and `label`; confirm labels are only `0` (Fake) and `1` (True). Check duplicate IDs within each split and intersections across official splits.
3. **Preserve splits:** Use `train_clean.csv` only for fitting; `validation_clean.csv` only for checkpoint/hyperparameter selection; keep `test_clean.csv` untouched until selection is complete. A subset must be deterministic and sampled within its official split.
4. **Image handling:** Match each image using the CSV `id`, verify it opens, and exclude failures. Never substitute, duplicate, or relabel an image. Save an ID/label/path manifest with split provenance.
5. **Environment:** Confirm Python 3.12 exact executable/version, required import checks, device availability, memory, and disk before training. Install only explicitly required packages from the relevant requirements file.
6. **Fit:** Record pretrained model/weights, seed, subset IDs/counts, transforms/tokenizer, loss and class weighting, optimizer, schedule, batch/accumulation, epoch limit, and device. Log errors explicitly.
7. **Validate/select:** Compute required metrics on validation after each eligible checkpoint. Apply the stated validation-based selection rule and record every evaluated validation epoch.
8. **Final test:** Load test examples only after checkpoint selection. Evaluate the selected checkpoint once for final reporting; do not tune based on test results.
9. **Persist:** Save the selected model under `models/<name>/`; write actual metrics, confusion-matrix label order, split/sample counts, configuration, selected epoch, and duration to a new `experiments/<name>_results.json`. Preserve previous results.
10. **Audit:** Parse the JSON, verify model files and manifest paths, reconcile recorded counts/IDs, and report tests, failures, and limitations. A configured script is not a completed experiment.
