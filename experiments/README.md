# Text Baseline Results

These results represent the project's text-only TF-IDF + Logistic Regression baseline evaluated on the Fakeddit processed validation and test splits. The confusion matrix is for the test split.

The recorded values are actual measured results provided for this project; they were not recalculated or estimated as part of this record.

## Transformer text-classification experiment

`src/text/distilbert_classifier.py` fine-tunes the BERT-family model `distilbert-base-uncased` using only the `text` and `label` columns. DistilBERT is selected because the available machine has no CUDA-capable GPU and limited free RAM; full BERT fine-tuning is not a practical default for this hardware.

By default, training and validation each use a deterministic, stratified 8,000-row subset (seed 42) of their own split. The test split is not loaded until training and validation-based checkpoint selection are complete, and the final evaluation uses the full test split once. Training uses class-weighted cross-entropy, a maximum sequence length of 128, batches of 4 with gradient accumulation of 4, up to 2 epochs, and early stopping based on validation binary F1.

Install the optional transformer dependencies from the project root with `python -m pip install -r requirements-transformer.txt`, then run `python src/text/distilbert_classifier.py`. The script saves the selected model and tokenizer to `models/distilbert_fakeddit_text/` and writes the actual validation history and final test metrics to `experiments/transformer_results.json` after a successful run. No transformer results are reported until that training and final evaluation have completed.

## ResNet-50 image-classification experiment

`src/image/resnet_classifier.py` prepares deterministic, label-stratified subsets independently from the official processed train, validation, and test CSVs. By default it targets 500 training, 150 validation, and 150 test images (seed 42); unavailable or unreadable images are excluded, and the actual usable IDs and counts are written to `data/images/resnet_subset/manifest.json`. Image files use the downloader convention `<id>.jpg`. Existing `data/images/train/` files may be reused only for matching official training IDs. The old `data/images/test_batch/` is not used because its IDs belong to the train split.

The experiment uses ImageNet V2 pretrained ResNet-50 weights and ImageNet normalization after resize to 256 and center crop to 224. Its backbone is frozen and only the new binary classification head is trained on CPU, with class-weighted cross-entropy. The checkpoint is selected using validation binary F1; test images are evaluated only after selection. On successful completion the model is saved in `models/resnet_fakeddit/` and actual validation history, final test metrics, counts, and duration are written to `experiments/resnet_results.json`. Install its dependencies with `python -m pip install -r requirements-image.txt`, then run `python src/image/resnet_classifier.py`. Results are not available until the script has completed successfully.
