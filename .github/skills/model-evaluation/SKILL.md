---
name: model-evaluation
description: Calculate and audit classification metrics, distinguish validation from test results, and compare Fakeddit experiments fairly.
---

# Classification Evaluation Procedure

1. Confirm label encoding and the positive class. For Fakeddit, use `0 = Fake`, `1 = True`; state `pos_label=1` for binary Precision, Recall, and F1.
2. Calculate from actual labels and predictions:
   - Accuracy
   - binary Precision, Recall, and F1 for positive label 1
   - Macro F1 across labels 0 and 1
   - confusion matrix with explicit `[0, 1]` label ordering (rows = actual, columns = predicted)
3. Record sample count, class counts, split name, metric averaging/zero-division policy, and matrix order beside metrics.
4. Treat validation as model/checkpoint/threshold selection data. Treat test as final held-out evaluation after choices are fixed. Never use test metrics to choose a model or hyperparameter.
5. For imbalanced data, report class distribution and whether loss weights or sampling were applied. Do not present accuracy alone as sufficient evidence.
6. Before comparing experiments, verify same official split, exact evaluated examples/subset, label mapping, preprocessing, positive-class definition, and metric averaging. If any differ, qualify or avoid direct ranking.
7. Preserve raw numeric values in JSON. Do not round for source data, infer missing metrics from others, fabricate confusion-matrix counts, or claim statistical significance without an appropriate analysis.
8. Interpret results cautiously and state limitations such as CPU-only training, subset size, unavailable inputs, or lack of repeated runs.
