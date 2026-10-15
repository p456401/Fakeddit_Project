"""Prepare official-split Fakeddit images and train a ResNet-50 classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILES = {
    "train": PROJECT_ROOT / "data" / "processed" / "train_clean.csv",
    "validation": PROJECT_ROOT / "data" / "processed" / "validation_clean.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "test_clean.csv",
}
IMAGE_ROOT = PROJECT_ROOT / "data" / "images" / "resnet_subset"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "resnet_fakeddit"
RESULTS_FILE = PROJECT_ROOT / "experiments" / "resnet_results.json"
LABELS = [0, 1]
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic official-split image subsets and train ResNet-50."
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=12.0)
    args = parser.parse_args()
    if min(
        args.train_size,
        args.validation_size,
        args.test_size,
        args.epochs,
        args.batch_size,
    ) < 1:
        parser.error("subset sizes, epochs, and batch size must be positive")
    if args.request_timeout <= 0:
        parser.error("request timeout must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_split(split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(SPLIT_FILES[split_name], usecols=["id", "image_url", "label"])
    if frame[["id", "image_url", "label"]].isna().any().any():
        raise ValueError(f"{SPLIT_FILES[split_name]} has a missing id, image_url, or label")
    frame["id"] = frame["id"].astype(str)
    if not frame["id"].str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError(f"{SPLIT_FILES[split_name]} contains an unsafe sample ID")
    frame["image_url"] = frame["image_url"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin(LABELS).all():
        raise ValueError(f"{SPLIT_FILES[split_name]} contains labels other than 0 and 1")
    if frame["id"].duplicated().any():
        raise ValueError(f"{SPLIT_FILES[split_name]} contains duplicate IDs")
    return frame


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def proportional_class_targets(frame: pd.DataFrame, subset_size: int) -> dict[int, int]:
    class_counts = frame["label"].value_counts().to_dict()
    if len(class_counts) != len(LABELS):
        raise ValueError("Each official split must contain both binary labels")
    if subset_size > len(frame):
        raise ValueError(
            f"Requested {subset_size} rows from a split with only {len(frame)} rows"
        )
    exact = {label: subset_size * class_counts[label] / len(frame) for label in LABELS}
    targets = {label: int(exact[label]) for label in LABELS}
    remaining = subset_size - sum(targets.values())
    for label in sorted(LABELS, key=lambda item: exact[item] - targets[item], reverse=True):
        if remaining == 0:
            break
        targets[label] += 1
        remaining -= 1
    if any(targets[label] == 0 for label in LABELS):
        raise ValueError("Each selected subset must include both labels")
    return targets


def cached_image_path(split_name: str, sample_id: str) -> Path | None:
    candidates = [IMAGE_ROOT / split_name / f"{sample_id}.jpg"]
    if split_name == "train":
        candidates.append(PROJECT_ROOT / "data" / "images" / "train" / f"{sample_id}.jpg")
    for path in candidates:
        if path.is_file() and valid_image(path):
            return path
    return None


def download_image(
    session: requests.Session,
    image_url: str,
    output_path: Path,
    timeout: float,
) -> tuple[bool, str]:
    if urlparse(image_url).scheme not in {"http", "https"}:
        return False, "unsupported URL scheme"
    temporary_path = output_path.with_suffix(".download")
    try:
        response = session.get(image_url, timeout=(5.0, timeout), stream=True)
        response.raise_for_status()
        with Image.open(response.raw) as image:
            image.convert("RGB").save(temporary_path, format="JPEG", quality=95)
        response.close()
        if not valid_image(temporary_path):
            temporary_path.unlink(missing_ok=True)
            return False, "downloaded file failed image verification"
        temporary_path.replace(output_path)
        return True, ""
    except (requests.RequestException, OSError, UnidentifiedImageError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def prepare_split(
    split_name: str,
    subset_size: int,
    seed: int,
    timeout: float,
    session: requests.Session,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_split(split_name)
    class_targets = proportional_class_targets(frame, subset_size)
    split_directory = IMAGE_ROOT / split_name
    split_directory.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    failed_attempts = 0
    attempted = 0
    available_cache_ids = {
        path.stem
        for path in (split_directory.glob("*.jpg") if split_directory.exists() else [])
    }
    if split_name == "train":
        available_cache_ids.update(
            path.stem
            for path in (PROJECT_ROOT / "data" / "images" / "train").glob("*.jpg")
        )

    for label in LABELS:
        class_frame = frame.loc[frame["label"] == label].sample(
            frac=1,
            random_state=seed + SPLIT_SEED_OFFSETS[split_name] * 100 + label,
        )
        max_attempts = max(class_targets[label] * 3, class_targets[label] + 25)
        cached_frame = class_frame.loc[class_frame["id"].isin(available_cache_ids)]
        pending_frame = class_frame.loc[
            ~class_frame["id"].isin(available_cache_ids)
        ].head(max_attempts)
        ordered_records = (
            cached_frame.to_dict(orient="records")
            + pending_frame.to_dict(orient="records")
        )
        class_selected = 0
        class_attempts = 0

        for record in ordered_records:
            if class_selected >= class_targets[label] or class_attempts >= max_attempts:
                break
            sample_id = record["id"]
            output_path = split_directory / f"{sample_id}.jpg"
            source = cached_image_path(split_name, sample_id)
            attempted += 1
            class_attempts += 1
            if source is not None:
                if source.resolve() != output_path.resolve():
                    shutil.copyfile(source, output_path)
                if not valid_image(output_path):
                    output_path.unlink(missing_ok=True)
                    source = None
                else:
                    selected.append(
                        {
                            "id": sample_id,
                            "label": label,
                            "image_path": str(output_path.relative_to(PROJECT_ROOT)),
                            "source": "existing verified image",
                        }
                    )
                    class_selected += 1
                    continue
            success, error = download_image(
                session,
                record["image_url"],
                output_path,
                timeout,
            )
            if success:
                selected.append(
                    {
                        "id": sample_id,
                        "label": label,
                        "image_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "source": "downloaded and verified",
                    }
                )
                class_selected += 1
            else:
                failed_attempts += 1
                print(
                    f"{split_name}: excluding unreadable/unavailable image "
                    f"id={sample_id} ({error})",
                    flush=True,
                )

        if class_selected < class_targets[label]:
            raise RuntimeError(
                f"{split_name} label {label}: found {class_selected} readable images, "
                f"needed {class_targets[label]} after {class_attempts} deterministic candidates"
            )

    selected_frame = pd.DataFrame(selected).sort_values(["label", "id"]).reset_index(drop=True)
    if selected_frame["id"].duplicated().any():
        raise RuntimeError(f"{split_name} selection contains duplicate sample IDs")
    counts = Counter(selected_frame["label"].tolist())
    summary = {
        "official_rows_available": len(frame),
        "target_images": subset_size,
        "usable_images": len(selected_frame),
        "usable_label_counts": {str(label): counts[label] for label in LABELS},
        "candidate_attempts": attempted,
        "failed_or_unreadable_attempts": failed_attempts,
        "verified_sample_ids": selected_frame["id"].tolist(),
    }
    return selected_frame, summary


class FakedditImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, frame: pd.DataFrame, image_transform: Any) -> None:
        self.frame = frame
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = PROJECT_ROOT / row["image_path"]
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = self.image_transform(rgb)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Prepared image became unreadable: {path}") from exc
        return tensor, int(row["label"])


def make_model(weights: ResNet50_Weights) -> nn.Module:
    model = resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def calculate_metrics(
    actual: list[int],
    predicted: list[int],
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())
    return calculate_metrics(actual, predicted)


def extract_frozen_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def train(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    set_seed(args.seed)
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print("Preparing deterministic image subsets from official CSV splits.", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Fakeddit-ResNet50-experiment/1.0"})
    subsets: dict[str, pd.DataFrame] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    requested = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    for split_name in ("train", "validation", "test"):
        subsets[split_name], split_summaries[split_name] = prepare_split(
            split_name,
            requested[split_name],
            args.seed,
            args.request_timeout,
            session,
        )
        print(
            f"{split_name}: prepared {len(subsets[split_name])} readable images "
            f"with labels {split_summaries[split_name]['usable_label_counts']}",
            flush=True,
        )
    session.close()

    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    (IMAGE_ROOT / "manifest.json").write_text(
        json.dumps(split_summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    weights = ResNet50_Weights.IMAGENET1K_V2
    weights_transform = weights.transforms()
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights_transform.mean,
                std=weights_transform.std,
            ),
        ]
    )
    train_dataset = FakedditImageDataset(subsets["train"], image_transform)
    validation_dataset = FakedditImageDataset(subsets["validation"], image_transform)
    test_dataset = FakedditImageDataset(subsets["test"], image_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = make_model(weights).to(device)
    label_counts = subsets["train"]["label"].value_counts()
    class_weights = torch.tensor(
        [
            len(subsets["train"]) / (len(LABELS) * label_counts[label])
            for label in LABELS
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    best_validation_f1 = -1.0
    best_epoch = 0
    validation_history: list[dict[str, Any]] = []
    checkpoint_path = MODEL_DIRECTORY / "resnet50_best.pth"
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.eval()
        model.fc.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = extract_frozen_features(model, images)
            logits = model.fc(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        validation_metrics = evaluate(model, validation_loader, device)
        validation_history.append({"epoch": epoch, **validation_metrics})
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={running_loss / len(train_loader):.4f}, "
            f"validation_f1={validation_metrics['f1']:.4f}, "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": "ResNet-50",
                    "weights": weights.name,
                    "num_classes": len(LABELS),
                    "label_mapping": {"0": "Fake", "1": "True"},
                    "best_epoch": best_epoch,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    if best_epoch == 0:
        raise RuntimeError("No validation-selected ResNet checkpoint was produced")
    training_duration_seconds = time.perf_counter() - training_started

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    experiment_duration_seconds = time.perf_counter() - started
    selected_validation_metrics = next(
        entry for entry in validation_history if entry["epoch"] == best_epoch
    )
    results = {
        "model_name": "ResNet-50",
        "pretrained_weights": weights.name,
        "dataset_name": "Fakeddit (image-only input)",
        "label_mapping": {"0": "Fake", "1": "True"},
        "selection_metric": "validation binary F1 (positive label 1)",
        "best_epoch": best_epoch,
        "epochs_completed": len(validation_history),
        "seed": args.seed,
        "device": str(device),
        "training_approach": (
            "Frozen pretrained ResNet-50 backbone; only the new binary "
            "fully connected classification head was trained."
        ),
        "preprocessing": {
            "resize_shorter_side": 256,
            "center_crop": 224,
            "normalization": "ImageNet pretrained-weight mean and standard deviation",
        },
        "configuration": {
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "class_weighted_cross_entropy": True,
            "backbone_frozen": True,
            "fine_tuned_backbone_blocks": [],
        },
        "splits": split_summaries,
        "validation_history": validation_history,
        "selected_validation_metrics": selected_validation_metrics,
        "test_metrics": test_metrics,
        "training_duration_seconds": training_duration_seconds,
        "experiment_duration_seconds": experiment_duration_seconds,
        "model_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Training duration seconds: {training_duration_seconds:.2f}", flush=True)
    print(f"Experiment duration seconds: {experiment_duration_seconds:.2f}", flush=True)
    print(f"Model saved: {checkpoint_path}", flush=True)
    print(f"Results saved: {RESULTS_FILE}", flush=True)
    print(json.dumps({"validation_history": validation_history, "test_metrics": test_metrics}, indent=2), flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(
            "Missing image-classification dependencies. Install them with "
            "'python -m pip install -r requirements-image.txt'.",
            file=sys.stderr,
        )
        raise exc
