---
name: run-experiment
description: Inspect, run, and record a requested Fakeddit experiment using actual measured results.
argument-hint: "[experiment name or script]"
agent: ML Engineer
---

Run the requested experiment: ${input:experiment:experiment name or script}

1. Inspect its configuration, required packages, input split files, existing outputs, and saved models.
2. Verify official split provenance and ID separation. Confirm Python 3.12 and imports using the exact interpreter that will run the experiment.
3. Preserve raw/processed data, existing model branches, and existing result files. Do not overwrite an existing result without explicit user authorization.
4. Summarize subset/configuration and any expensive downloads or training before starting. Do not broaden the requested scope.
5. Run the actual training/evaluation only after the requested run is clear. Select checkpoints using validation data; use test data only once for the selected model's final evaluation.
6. Save only actual computed metrics and reproducibility metadata. Report exact paths, configuration, duration, sample counts, errors, and any missing metrics. Never fabricate or estimate results.
