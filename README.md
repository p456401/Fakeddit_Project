# Context-Aware Multimodal Misinformation Verification

An ML-only B.Tech major-project repository for experiments on Fakeddit. The current implementation includes evaluated text-only and image-only classifiers and an exploratory, validation-selected late-fusion baseline measured on a small aligned image/text subset. It is **not yet an end-to-end context-aware multimodal verifier**: CLIP and context metadata analysis remain incomplete.

No frontend is required for this submission.

## What is implemented

| Branch | Inputs | Saved evidence | Test scope |
|---|---|---|---|
| TF-IDF + Logistic Regression | Text | `experiments/baseline_results.json`; vectorizer and classifier under `models/` | 59,163 official test rows |
| DistilBERT (`distilbert-base-uncased`) | Text | `experiments/transformer_results.json`; local model and tokenizer under `models/distilbert_fakeddit_text/` | 59,163 official test rows |
| ResNet-50 (ImageNet V2 initialization) | Image | `experiments/resnet_results.json`; checkpoint under `models/resnet_fakeddit/` | Held-out subset of 150 official test images |
| Text-image late fusion | Text + aligned image | `experiments/multimodal_late_fusion_20261015_140619.json` | Exploratory; 150 aligned validation and 150 held-out test examples only |
| CLIP consistency | Text + image | No local CLIP weights; no download performed | Future work |
| Context metadata model | Domain/subreddit fields exist in raw TSVs but are not in the processed classifier inputs | Not implemented/evaluated | Future work |

Prediction labels are `0 = Fake` and `1 = True`. Binary Precision, Recall, and F1 use `pos_label=1` (True). The saved TF-IDF full-test report does not include Macro F1; none is inferred here.

## Recorded results

Values below are transcribed from the saved result JSON files; they are not newly estimated. The three test evaluations use different scopes/modalities and should not be treated as a controlled ranking.

| Model | Test accuracy | Precision (True) | Recall (True) | F1 (True) | Macro F1 | Test sample scope |
|---|---:|---:|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 83.52% | 76.28% | 84.66% | 80.25% | Not recorded | 59,163 rows |
| DistilBERT | 83.35% | 76.55% | 83.48% | 79.87% | 82.84% | 59,163 rows |
| ResNet-50 | 78.67% | 70.15% | 79.66% | 74.60% | 78.11% | 150 held-out images |

The processed training split contains 341,460 Fake and 221,006 True examples (562,466 total); the validation split contains 35,974 Fake and 23,195 True examples (59,169); the test split contains 35,763 Fake and 23,400 True examples (59,163). Class weighting is recorded for each trained classifier. The baseline and DistilBERT test matrices are for the complete test split; ResNet and fusion metrics below use a 150-image aligned subset. Split-ID checks found no within-split duplicates or pairwise ID overlap; that does not rule out semantically related examples with different IDs.

### Confusion matrices

Rows are actual labels and columns are predictions, in label order `[0 (Fake), 1 (True)]`.

| Model | Test confusion matrix |
|---|---|
| TF-IDF | `[[29603, 6160], [3589, 19811]]` |
| DistilBERT | `[[29780, 5983], [3865, 19535]]` |
| ResNet-50 (150 images) | `[[71, 20], [12, 47]]` |

### Exploratory aligned late fusion

The new [fusion result JSON](experiments/multimodal_late_fusion_20261015_140619.json) pairs text and image probabilities by sample ID from the existing official image manifest. It evaluates 150 validation and 150 held-out test examples (91 Fake, 59 True in each); the remaining 59,013 official test rows are not represented in this result. It uses probability averaging, `p_true = alpha * p_true_text + (1-alpha) * p_true_resnet`, with a 0.5 decision threshold. Alpha is selected on validation only by positive-class binary F1, searching 0.00-1.00 in 0.01 increments.

| Aligned-subset model | Validation alpha | Test accuracy | Precision (True) | Recall (True) | F1 (True) | Macro F1 | Test confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + ResNet-50 | 0.63 | 88.00% | 80.60% | 91.53% | 85.71% | 87.68% | `[[78, 13], [5, 54]]` |
| DistilBERT + ResNet-50 | 0.38 | 86.00% | 78.79% | 88.14% | 83.20% | 85.60% | `[[77, 14], [7, 52]]` |

These are exploratory subset measurements, not full-test performance estimates. The validation tuning sample is small; no claim of statistical significance is made. Each fusion result and its component classifiers are evaluated on the same aligned rows, but the standalone full-test numbers above use a different evaluation scope.

## Setup

Use Python **3.12**. Create a fresh environment instead of modifying a working environment or the project's old virtual environments:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements include the transformer and image branches and may download sizable ML packages. If only text baseline inference is needed, install `numpy`, `pandas`, `scikit-learn`, and `joblib` in the environment instead of installing the full ML stack. No dataset is bundled. The `.pth` and `.safetensors` checkpoints are tracked with Git LFS; a clone that skips LFS downloads cannot load those weights.

## Dataset

Obtain Fakeddit using its official distribution and follow its terms. Keep raw and processed datasets out of Git. Place the official TSV inputs under `data/raw/` and run the preprocessing script only when preparing the local dataset:

```powershell
.\.venv\Scripts\python.exe src\data_preprocessing.py
```

The experiment scripts consume `data/processed/train_clean.csv`, `validation_clean.csv`, and `test_clean.csv`. Preserve these official splits. Do not tune on test data. Image experiments join by sample ID and exclude unavailable/unreadable images; never substitute images. See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [experiments/README.md](experiments/README.md) for local artifact and experiment details.

## Demo and experiments

After cloning with Git LFS and installing dependencies, run the local model demo:

```powershell
.\.venv\Scripts\python.exe src\multimodal_inference.py predict --text "Example claim text"
.\.venv\Scripts\python.exe src\multimodal_inference.py predict --text "Example claim text" --image "C:\path\to\local-image.jpg"
```

The CLI reports each available model separately; its example output is an inference demonstration, **not** an evaluation result. Check `.\.venv\Scripts\python.exe src\multimodal_inference.py predict --help` for options. Do not run training just to test inference.

With the processed official splits and manifest-listed images present, the repeatable validation-selected exploratory evaluation command is:

```powershell
.\.venv\Scripts\python.exe src\multimodal_inference.py evaluate-fusion
```

It writes to a timestamped new result JSON and refuses to overwrite a requested existing output. It does not train a model or download an image/checkpoint.

Run focused tests and regenerate figures/report with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\generate_figures.py
.\.venv\Scripts\python.exe tools\build_report.py
```

Existing experiment scripts are:

```powershell
.\.venv\Scripts\python.exe src\text_baseline.py
.\.venv\Scripts\python.exe src\text\distilbert_classifier.py
.\.venv\Scripts\python.exe src\image\resnet_classifier.py
```

Training writes model/results artifacts. Inspect each script's output paths first and never overwrite recorded results unless intentionally authorized. The original result files are retained unchanged.

## Report

- [Project report (DOCX)](docs/Project_Report.docx)
- [Project report (PDF)](docs/Project_Report.pdf)
- [Experiment details](experiments/README.md)

The report states the measured scopes and limitations. Team and institutional details are omitted because they were not verified in the workspace.

## Limitations and next work

- Late fusion is implemented only as an exploratory aligned-subset baseline. It has not been evaluated across all text examples, nor does it use a separate context metadata model.
- ResNet results are from 150 held-out images, not all 59,163 test rows.
- DistilBERT trained on deterministic 8,000-row train and validation subsets. Existing result records report CPU execution.
- CLIP consistency and context metadata analysis need implementation/evaluation before claiming those components are complete.
- Text/image ID non-overlap does not prove semantic or near-duplicate independence.
- This repository does not implement a frontend, backend service or deployment.

## References

1. S. Nakamura, K. Levy, and W. Y. Wang, “r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection,” arXiv:1911.03854, 2019. https://arxiv.org/abs/1911.03854
2. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” NAACL-HLT, 2019. https://arxiv.org/abs/1810.04805
3. V. Sanh et al., “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” NeurIPS EMC2 Workshop, 2019. https://arxiv.org/abs/1910.01108
4. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” CVPR, 2016. https://arxiv.org/abs/1512.03385
5. A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML, 2021. https://arxiv.org/abs/2103.00020
6. C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008. https://nlp.stanford.edu/IR-book/
