# Context-Aware Multimodal Misinformation Verification System

## B.Tech Major Project — Machine-Learning Submission

**Project team / institute:** Not included; verified details were not present in the inspected workspace.
**Submission scope:** ML experiments, reproducible inference and evaluation documentation. A frontend is not required.

## Abstract

This project investigates misinformation classification on Fakeddit using text and image modalities. The verified implementation contains a TF-IDF plus Logistic Regression text baseline, a fine-tuned DistilBERT text classifier, and a ResNet-50 image classifier. Saved result files report full official test-split evaluation for the two text systems and held-out subset evaluation for the image model. Existing artifacts and scripts are inspected; TF-IDF and DistilBERT were additionally smoke-tested for local inference. The current models remain independent classifiers. This report does not claim that context-aware multimodal fusion or CLIP has been completed. Results, sample scopes, class imbalance, and limits to comparability are made explicit.

## 1. Problem and scope

Misinformation can be expressed in text and visual content. Fakeddit is a multimodal fake-news benchmark built from Reddit content [1]. This project evaluates practical text and image classifiers on the project's processed Fakeddit split files. Labels are `0 = Fake` and `1 = True`.

The submission focuses on a small, reproducible ML deliverable rather than a user interface. A context-aware fused verifier would additionally require an implemented, leakage-controlled combination of aligned modalities and any context fields. Existing standalone results do not establish that such a system has been completed.

## 2. Dataset and split protocol

The workspace contains official train, validation and test inputs transformed into `train_clean.csv`, `validation_clean.csv` and `test_clean.csv`. Processed row totals and label counts are:

| Official split | Total | Fake (0) | True (1) | Fake share |
|---|---:|---:|---:|---:|
| Train | 562,466 | 341,460 | 221,006 | 60.71% |
| Validation | 59,169 | 35,974 | 23,195 | 60.80% |
| Test | 59,163 | 35,763 | 23,400 | 60.44% |

The data are imbalanced toward Fake, so accuracy alone is insufficient. The experiment implementations use class-weighted losses/class weights. Binary Precision, Recall and F1 refer to positive label 1 (True). Confusion matrices use `[0,1]` ordering with actual labels in rows and predictions in columns.

The inspected split files had no null/duplicate IDs within each split and no shared IDs across train, validation and test. This is only an ID-level check; it cannot rule out related examples or semantic/near-duplicate content under different IDs. Image samples are joined by exact IDs, and the ResNet evaluation uses only readable images in its saved official-split manifest.

Validation data are used for checkpoint selection. The saved DistilBERT and ResNet results report selection by validation binary F1. The test split is held out for final evaluation. The ResNet test numbers below apply to its manifest's 150-image subset, not the full test split.

## 3. Model methods

### 3.1 TF-IDF + Logistic Regression

The baseline uses text unigrams and bigrams with a 50,000-feature cap and sublinear term frequency, followed by Logistic Regression (`max_iter=1000`, `class_weight="balanced"`). The source script fits the TF-IDF vocabulary on training text and transforms validation/test text with the fitted vectorizer. Saved artifacts are `models/tfidf_vectorizer.pkl` and `models/logistic_regression.pkl`.

### 3.2 DistilBERT

The text transformer is `distilbert-base-uncased`, a compact distilled Transformer derived from BERT [2,3]. The recorded run used seed 42; deterministic stratified 8,000-row train and 8,000-row validation subsets; full test split (59,163 rows); maximum sequence length 128; batch size 4 with gradient accumulation 4; up to two epochs; learning rate 2e-5; weight decay 0.01; CPU execution; and class-weighted cross-entropy. Epoch 2 was selected by validation binary F1. The saved model/tokenizer and result JSON are present.

### 3.3 ResNet-50

The image branch uses ResNet-50 [4] initialized with ImageNet `IMAGENET1K_V2` weights, with the backbone frozen and a new binary classification head trained. The run used CPU, seed 42, batch size 8, AdamW learning rate 0.001, weight decay 0.0001, weighted cross-entropy, and two epochs. Image processing resized the shorter side to 256, center-cropped to 224 and applied ImageNet normalization. The deterministic manifest records 500 usable training images, 150 validation images and 150 test images, sampled within the corresponding official splits. Epoch 2 was selected on validation F1. The final test evaluation is therefore a small, held-out image subset.

### 3.4 Exploratory text-image late fusion

The separate baseline result files do not contain fusion measurements. A new exploratory evaluation, saved as `experiments/multimodal_late_fusion_20261015_140619.json`, aligns the model inputs with the existing ID manifest and official validation/test CSVs. The validation and test subsets each contain 150 examples (91 Fake and 59 True), and IDs are checked against their official split. The experiment computes class probabilities from TF-IDF, DistilBERT and ResNet-50. For each text model independently, it chooses alpha on validation from 0.00 through 1.00 in 0.01 increments to maximize binary F1 for label 1; ties are resolved by higher accuracy, then proximity to 0.50, then lower alpha. The selected fusion is `p_true = alpha * p_true_text + (1-alpha) * p_true_resnet`, with a 0.5 threshold. The selected alpha is fixed before test data are read and evaluated.

The validation-selected alpha values are 0.63 for TF-IDF plus ResNet and 0.38 for DistilBERT plus ResNet. Test metrics on the 150-image aligned subset are:

| Aligned-subset system | Accuracy | Precision (True) | Recall (True) | F1 (True) | Macro F1 | Confusion matrix `[0,1]` |
|---|---:|---:|---:|---:|---:|---|
| TF-IDF + ResNet-50 (`alpha=0.63`) | 88.00% | 80.60% | 91.53% | 85.71% | 87.68% | `[[78, 13], [5, 54]]` |
| DistilBERT + ResNet-50 (`alpha=0.38`) | 86.00% | 78.79% | 88.14% | 83.20% | 85.60% | `[[77, 14], [7, 52]]` |

These exploratory measurements are not full-test estimates: only 150 of 59,163 official test rows (59,013 fewer) are included. The small validation subset makes the selected weights and metrics uncertain. The fusion results do not replace the full-split text evaluations or establish a general improvement. No statistical significance claim is made.

### 3.5 Context metadata and CLIP

The raw TSV headers include `domain` and `subreddit` metadata, but the processed classifier inputs contain only ID, text and label. Context metadata were not added to a model; community/domain signals can encode label correlations and require a separate leakage audit. No CLIP pretrained weights were present in the inspected artifacts/cache. The installed Transformers package contains CLIP model code, but the expensive checkpoint was not downloaded. CLIP and context modeling are therefore incomplete and have no reported metrics.

## 4. Evaluation results

All values below are transcribed from the existing result JSONs. Precision/Recall/F1 are binary scores for True (label 1). No missing metric is reconstructed.

| Model | Test accuracy | Precision | Recall | F1 | Macro F1 | Test examples |
|---|---:|---:|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 83.52% | 76.28% | 84.66% | 80.25% | Not recorded | 59,163 |
| DistilBERT | 83.35% | 76.55% | 83.48% | 79.87% | 82.84% | 59,163 |
| ResNet-50 | 78.67% | 70.15% | 79.66% | 74.60% | 78.11% | 150 images |
| TF-IDF + ResNet-50 fusion | 88.00% | 80.60% | 91.53% | 85.71% | 87.68% | 150 aligned test examples |
| DistilBERT + ResNet-50 fusion | 86.00% | 78.79% | 88.14% | 83.20% | 85.60% | 150 aligned test examples |

Test confusion matrices (rows = actual; columns = predicted; class order `[Fake, True]`):

| Model | Confusion matrix |
|---|---|
| TF-IDF | `[[29603, 6160], [3589, 19811]]` |
| DistilBERT | `[[29780, 5983], [3865, 19535]]` |
| ResNet-50 | `[[71, 20], [12, 47]]` |
| TF-IDF + ResNet-50 fusion (`alpha=0.63`) | `[[78, 13], [5, 54]]` |
| DistilBERT + ResNet-50 fusion (`alpha=0.38`) | `[[77, 14], [7, 52]]` |

Text branch comparisons use the same full test split and label convention, though their training protocols differ (the baseline's full training set versus DistilBERT's 8,000-row subset). ResNet and both fusion systems are evaluated on only 150 aligned images/examples, not a full-test comparison to text. Standalone classifier metrics within the aligned-subset fusion JSON differ from the independently recorded full-set metrics and should not be confused. The result JSON for the full-test TF-IDF run does not record Macro F1.

### Result plots

The comparison chart shows the recorded scalar metrics without suggesting equivalent test protocols. The confusion matrices show each model's actual test scope.

![Recorded test metric comparison](figures/model_comparison.png)

![Recorded test confusion matrices](figures/confusion_matrices.png)

## 5. Inference demonstration

The local inference CLI loads available model artifacts and returns individual branch predictions. Optional image inference requires a readable local image and available ResNet checkpoint. Predictions on demonstration inputs are not measured evaluation metrics. For repeatability, use the CLI command from the repository README and record the exact input/artifact version when presenting a demo.

TF-IDF and DistilBERT were verified to load and predict on two synthetic smoke-test strings using system Python 3.12.6. This checks the local inference path only; it does not evaluate accuracy or validate the ResNet inference environment. A fully clean install of all ML dependencies into a fresh environment was not performed.

## 6. Reproducibility and artifacts

The experiment result JSONs remain the primary numeric evidence: `experiments/baseline_results.json`, `experiments/transformer_results.json`, and `experiments/resnet_results.json`. Original result artifacts are preserved; any late-fusion result must use a separate file. The existing checkpoints are saved under `models/`. Fakeddit data are not included in the public repository; download and prepare data according to official distribution conditions.

Use Python 3.12 and install only the requirements needed for the desired branch. Keep raw data immutable, preserve official processed splits, fit learned preprocessing on train only, select checkpoints on validation, and reserve test data for final evaluation. The inference example is a smoke test; experiment training commands should only be run intentionally because they may use CPU time and create artifacts.

## 7. Limitations and future work

1. The exploratory fusion baseline is trained/tuned only by probability-weight selection on a 150-example validation subset; a complete context-aware verifier is not demonstrated.
2. ResNet and fusion test results use only 150 examples; their scope and uncertainty differ from full-test text results.
3. DistilBERT's training and validation subsets contain 8,000 examples each; results reflect that training budget.
4. ID-level checks do not prove semantic independence. More robust near-duplicate checks would require a separate, carefully designed audit.
5. Raw `domain` and `subreddit` metadata are available but were not used; community signals require a controlled feature and leakage analysis.
6. CLIP requires selecting and obtaining an appropriate pretrained checkpoint and then evaluating a clearly defined consistency method. It remains future work; no weights were downloaded.
7. No frontend, backend, deployment, or end-to-end verification service is implemented or required by the project's stated submission scope.
8. Runtime dependency installation and full retraining have not been claimed by this report.

## References

1. S. Nakamura, K. Levy, and W. Y. Wang. “r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection.” arXiv:1911.03854, 2019. https://arxiv.org/abs/1911.03854
2. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” NAACL-HLT, 2019. https://arxiv.org/abs/1810.04805
3. V. Sanh et al. “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.” NeurIPS EMC2 Workshop, 2019. https://arxiv.org/abs/1910.01108
4. K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition.” CVPR, 2016. https://arxiv.org/abs/1512.03385
5. A. Radford et al. “Learning Transferable Visual Models From Natural Language Supervision.” ICML, 2021. https://arxiv.org/abs/2103.00020
6. C. D. Manning, P. Raghavan, and H. Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008. https://nlp.stanford.edu/IR-book/
