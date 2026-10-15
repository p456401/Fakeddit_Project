---
name: review-results
description: Audit recorded Fakeddit experiment outputs and produce a leakage-aware academic interpretation.
argument-hint: "[result JSON or experiment]"
agent: Research Reviewer
---

Review the existing results for: ${input:results:result JSON or experiment}

- Inspect the result JSON, associated training/evaluation code, saved artifact presence, and relevant split metadata.
- Verify metric values from the recorded file; do not retrain or recompute on test data.
- Keep validation used for selection distinct from final test evaluation. Check for test-based tuning, split contamination, ID mismatch, duplicate leakage, and subset limitations.
- Compare metrics only when datasets, subsets, labels, and evaluation definitions are comparable.
- Report actual values and matrix label order; identify missing or inconsistent fields and claims unsupported by artifacts.
- Give an academically cautious interpretation. Do not invent results, imply causation, or make unsupported novelty claims.
