# Fakeddit Project Status

Checkpoint updated 2026-10-15 from the current repository artifacts. Metrics below are transcribed from the experiment result files and are not estimates.

## Dataset

- Raw official splits are `data/raw/multimodal_train.tsv`, `multimodal_validate.tsv`, and `multimodal_test_public.tsv`. `src/data_preprocessing.py` removes rows missing `image_url` and duplicate `id` values, selects the project columns, and renames `clean_title` to `text` and `2_way_label` to `label`.
- Processed split sizes and observed label counts:

  | Split | Rows | Label 0 (Fake) | Label 1 (True) |
  |---|---:|---:|---:|
  | Train | 562,466 | 341,460 | 221,006 |
  | Validation | 59,169 | 35,974 | 23,195 |
  | Test | 59,163 | 35,763 | 23,400 |

- Labels in the processed splits are binary (0 and 1); the project/model mapping is 0 = Fake, 1 = True.
- An identifier-level leakage check found no duplicate IDs within a split and zero ID overlap between each pair of train, validation, and test. This check does not establish absence of duplicate or related content under different IDs.

## Image download and validation

- `src/image_downloader.py` names downloaded files `data/images/train/<id>.jpg` and was configured for the first 1,000 training rows.
- The current `data/images/train/` contains 942 images. All 942 opened and passed PIL image verification; all correspond to IDs in the first 1,000 official training rows. `data/images/failed_downloads.csv` records 58 failures; failed images are not present as training examples.
- `data/images/test_batch/` contains 94 readable images, but their IDs belong to the official training split. They are not a held-out test set. `src/validate_images.py` validates this directory.
- The ResNet experiment uses separate prepared image subsets in `data/images/resnet_subset/{train,validation,test}/`, with sample IDs recorded in `data/images/resnet_subset/manifest.json`. Every selected ID belongs to its corresponding official CSV split, the three manifest ID sets have no overlap, and every selected image file is present.

## Completed experiments

### TF-IDF + Logistic Regression

The text-only baseline uses `src/text_baseline.py`; saved artifacts are `models/tfidf_vectorizer.pkl` and `models/logistic_regression.pkl`. Results below are recorded in `experiments/baseline_results.json`.

| Split | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation | 0.8321756325102672 | 0.7578231292517007 | 0.84048286265859236 | 0.7970155355682748 |
| Test | 0.8352179571691767 | 0.7628123676408302 | 0.8466239316239316 | 0.8025359016426648 |

Test confusion matrix (label order 0, 1): `[[29603, 6160], [3589, 19811]]`. Macro F1 is not recorded for this baseline.

### DistilBERT

`src/text/distilbert_classifier.py` fine-tuned `distilbert-base-uncased` on text. The saved model and tokenizer are in `models/distilbert_fakeddit_text/`; actual validation history and final test metrics are in `experiments/transformer_results.json`.

- Deterministic stratified subsets: 8,000 training rows and 8,000 validation rows (seed 42); full test split: 59,163 rows.
- Two epochs completed; epoch 2 was selected by validation binary F1. The run status record reports 97,782.51 seconds (27 hours, 9 minutes, 42.51 seconds).
- Selected validation metrics: Accuracy 0.82825; Precision 0.7531609195402299; Recall 0.8357780612244898; F1 0.7923216444981862; Macro F1 0.8229511035790164.
- Final test metrics: Accuracy 0.8335446140324189; Precision 0.7655380515714397; Recall 0.8348290598290599; F1 0.798683511181978; Macro F1 0.8283989247933864.
- Test confusion matrix (label order 0, 1): `[[29780, 5983], [3865, 19535]]`.
- The training/evaluation code selects the checkpoint using validation F1 and loads the test split only afterward for final evaluation.

### ResNet-50

`src/image/resnet_classifier.py` implements and has completed a ResNet-50 image-classification experiment. The trained checkpoint is `models/resnet_fakeddit/resnet50_best.pth`; the recorded results are in `experiments/resnet_results.json`.

- The experiment used 500 images from the official train split, 150 from the official validation split, and a held-out 150-image subset from the official test split. Label counts were respectively 304/196, 91/59, and 91/59 for labels 0/1. Five validation and four test candidates failed or were unreadable and were excluded.
- Architecture and initialization: ResNet-50 with ImageNet `IMAGENET1K_V2` pretrained weights. The backbone was frozen; only the new binary fully connected classification head was trained. Preprocessing resized the shorter side to 256, center-cropped to 224, and used the pretrained ImageNet normalization.
- Training used CPU, seed 42, batch size 8, AdamW learning rate 0.001, weight decay 0.0001, and class-weighted cross-entropy. Two epochs completed; epoch 2 was selected using validation binary F1.
- Selected validation metrics: Accuracy 0.6466666666666666; Precision 0.5454545454545454; Recall 0.6101694915254238; F1 0.576; Macro F1 0.6365714285714286.
- Validation confusion matrix (label order 0, 1): `[[61, 30], [23, 36]]`.
- Final held-out test-subset metrics: Accuracy 0.7866666666666666; Precision 0.7014925373134329; Recall 0.7966101694915254; F1 0.746031746031746; Macro F1 0.7810618500273673.
- Test confusion matrix (label order 0, 1): `[[71, 20], [12, 47]]`.
- Recorded training duration: 175.4954274000047 seconds; total experiment duration: 524.4290991000016 seconds.
- The test results are from the 150-image held-out subset, **not** the full Fakeddit test set of 59,163 rows.

### Exploratory aligned text-image late fusion

`src/multimodal_inference.py` provides local prediction for TF-IDF and DistilBERT text branches and optional local-image ResNet prediction. Its `evaluate-fusion` command adds a probability-averaging late-fusion baseline, with no model training or image download. Results are written to a new timestamped result file; the completed evaluation is `experiments/multimodal_late_fusion_20261015_140619.json`.

- Validation and test inputs were joined by sample ID from the existing image manifest to their corresponding official processed CSV split. The evaluated subsets contain 150 validation and 150 test examples (91 label 0, 59 label 1 per split); these are not full-split evaluations.
- For TF-IDF + ResNet, validation-only search selected text probability weight alpha 0.63; test-subset accuracy 0.88, positive-label-1 precision 0.8059701492537313, recall 0.9152542372881356, F1 0.8571428571428571, Macro F1 0.8768472906403941; confusion matrix `[[78, 13], [5, 54]]`.
- For DistilBERT + ResNet, validation-only search selected alpha 0.38; test-subset accuracy 0.86, positive-label-1 precision 0.7878787878787878, recall 0.8813559322033898, F1 0.832, Macro F1 0.856; confusion matrix `[[77, 14], [7, 52]]`.
- Fusion is the weighted average of positive-class probabilities, thresholded at 0.5. The alpha grid ran from 0.00 to 1.00 in 0.01 increments, maximizing validation binary F1 for label 1; deterministic tie breaks use accuracy, closeness to 0.5, then lower alpha. The test split was read only after alpha selection.
- These exploratory 150-example measurements are not full-test performance estimates. The selected weights may be unstable on a small validation subset, and no statistical significance claim is made.

## Project status and future work

- `data/`: raw TSVs, processed CSV splits, and image directories (`images/train`, `images/test_batch`).
- `src/`: dataset preprocessing and image utilities, TF-IDF baseline, `text/distilbert_classifier.py`, `image/resnet_classifier.py`, and `multimodal_inference.py`.
- `models/`: saved TF-IDF artifacts, DistilBERT model/tokenizer, and ResNet-50 checkpoint.
- `experiments/`: baseline, transformer, ResNet, and exploratory late-fusion result JSONs plus experiment notes.
- `requirements-transformer.txt` and `requirements-image.txt`: optional dependencies for the corresponding model branches.
- `app.py`: Streamlit application for local text/image inference and browsing saved results. The `app/` and `notebooks/` directories have no project files.

| Component | Status |
|---|---|
| Dataset preparation | COMPLETED |
| TF-IDF + Logistic Regression | COMPLETED |
| DistilBERT | COMPLETED |
| ResNet-50 | COMPLETED |
| Local inference CLI | COMPLETED — TF-IDF/DistilBERT plus optional local-image ResNet |
| Streamlit application | COMPLETED — local input, available-model predictions, saved metrics and confusion matrices |
| Text-image late fusion | PARTIAL — exploratory probability fusion evaluated on 150 aligned validation and 150 test examples |
| CLIP image-text consistency | NOT STARTED — future work |
| Context metadata analysis | NOT STARTED — raw `domain` and `subreddit` are available but not modeled |
| Full-split multimodal evaluation | NOT STARTED — future work |
| Dedicated URL extraction | NOT STARTED — future work |
| Backend/deployment | NOT STARTED — future work |

The current implementation consists of separate text and image branches, an exploratory late-fusion baseline limited to aligned image subsets, and a Streamlit interface for local inference and saved-result inspection. CLIP image-text consistency, context analysis, full-split multimodal evaluation, and dedicated URL extraction remain future work. The project is not yet an end-to-end context-aware multimodal verification system; no backend or deployment is implemented.
