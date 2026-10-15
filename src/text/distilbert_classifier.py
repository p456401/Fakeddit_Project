"""Fine-tune DistilBERT for Fakeddit text classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "validation_clean.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test_clean.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "distilbert_fakeddit_text"
RESULTS_FILE = PROJECT_ROOT / "experiments" / "transformer_results.json"
MODEL_NAME = "distilbert-base-uncased"
LABELS = [0, 1]


class TokenizedTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels.to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def collate_batch(
    features: list[dict[str, torch.Tensor]],
    tokenizer: Any,
) -> dict[str, torch.Tensor]:
    labels = torch.stack([feature.pop("labels") for feature in features])
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on a deterministic Fakeddit text subset."
    )
    parser.add_argument("--train-limit", type=int, default=8_000)
    parser.add_argument("--validation-limit", type=int, default=8_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.train_limit,
        args.validation_limit,
        args.max_length,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.patience,
    ) < 1:
        parser.error("limits, sizes, epochs, and patience must be positive integers")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["text", "label"])
    if frame["label"].isna().any():
        raise ValueError(f"{path} contains missing labels")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin(LABELS).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame.reset_index(drop=True)


def stratified_subset(
    frame: pd.DataFrame,
    limit: int,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    if limit >= len(frame):
        return frame
    if limit < len(LABELS):
        raise ValueError(f"{split_name} limit must be at least {len(LABELS)}")
    subset, _ = train_test_split(
        frame,
        train_size=limit,
        random_state=seed,
        stratify=frame["label"],
    )
    return subset.reset_index(drop=True)


def make_dataloader(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = TokenizedTextDataset(frame["text"], frame["label"], tokenizer, max_length)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=lambda features: collate_batch(features, tokenizer),
        num_workers=0,
    )


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, Any], list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    actual: list[int] = []
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())
    metrics = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(
            precision_score(actual, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(actual, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(actual, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(actual, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            actual, predictions, labels=LABELS
        ).tolist(),
    }
    return metrics, actual, predictions


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading training and validation data; test data remains untouched.")

    train_frame = stratified_subset(
        load_split(TRAIN_FILE), args.train_limit, args.seed, "training"
    )
    validation_frame = stratified_subset(
        load_split(VALIDATION_FILE),
        args.validation_limit,
        args.seed,
        "validation",
    )
    print(f"Training rows used: {len(train_frame):,}")
    print(f"Validation rows used: {len(validation_frame):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Fake", 1: "True"},
        label2id={"Fake": 0, "True": 1},
    ).to(device)

    train_loader = make_dataloader(
        train_frame,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = make_dataloader(
        validation_frame,
        tokenizer,
        args.max_length,
        max(args.batch_size, 8),
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(LABELS),
        y=train_frame["label"].to_numpy(),
    )
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = (
        len(train_loader) + args.gradient_accumulation_steps - 1
    ) // args.gradient_accumulation_steps
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_updates,
    )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    best_validation_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    validation_history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)

        for batch_index, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=loss_weights,
            )
            total_loss += loss.item()

            window_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            window_size = min(
                args.gradient_accumulation_steps,
                total_batches - window_start,
            )
            (loss / window_size).backward()

            at_window_end = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            )
            if at_window_end:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_metrics, _, _ = evaluate(model, validation_loader, device)
        validation_history.append({"epoch": epoch, **validation_metrics})
        mean_loss = total_loss / total_batches
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={mean_loss:.4f}, "
            f"validation_f1={validation_metrics['f1']:.4f}, "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(MODEL_DIRECTORY)
            tokenizer.save_pretrained(MODEL_DIRECTORY)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation F1 did not improve.")
                break

    if best_epoch == 0:
        raise RuntimeError("Training completed without selecting a validation checkpoint")

    del model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIRECTORY).to(device)
    model.eval()

    print("Loading test data for the selected checkpoint's final evaluation.")
    test_frame = load_split(TEST_FILE)
    test_loader = make_dataloader(
        test_frame,
        tokenizer,
        args.max_length,
        max(args.batch_size, 8),
    )
    test_metrics, _, _ = evaluate(model, test_loader, device)

    results = {
        "model_name": "DistilBERT",
        "pretrained_model": MODEL_NAME,
        "dataset_name": "Fakeddit multimodal dataset (text-only input)",
        "label_mapping": {"0": "Fake", "1": "True"},
        "selection_metric": "validation binary F1 (positive label 1)",
        "best_epoch": best_epoch,
        "training_rows_available": len(load_split(TRAIN_FILE)),
        "training_rows_used": len(train_frame),
        "validation_rows_available": len(load_split(VALIDATION_FILE)),
        "validation_rows_used": len(validation_frame),
        "test_rows_evaluated": len(test_frame),
        "seed": args.seed,
        "configuration": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size
            * args.gradient_accumulation_steps,
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "class_weighted_loss": True,
            "device": str(device),
        },
        "validation_history": validation_history,
        "test_metrics": test_metrics,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Selected model and tokenizer saved to {MODEL_DIRECTORY}")
    print(f"Final test results saved to {RESULTS_FILE}")
    print(json.dumps(test_metrics, indent=2))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is unavailable; training and evaluation will run on CPU.")
    train(parse_args())


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(
            "Missing transformer dependencies. Install them with "
            "'python -m pip install -r requirements-transformer.txt'.",
            file=sys.stderr,
        )
        raise exc
