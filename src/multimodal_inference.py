"""Local inference and validation-selected late fusion for saved Fakeddit models.

All model assets are loaded from this workspace. This module never downloads
model weights, fetches images, or modifies existing artifacts/results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_ROOT = PROJECT_ROOT / "data" / "images" / "resnet_subset"
MANIFEST_PATH = IMAGE_ROOT / "manifest.json"
TFIDF_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
LOGISTIC_REGRESSION_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"
DISTILBERT_DIR = PROJECT_ROOT / "models" / "distilbert_fakeddit_text"
RESNET_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "resnet_fakeddit" / "resnet50_best.pth"
)
SPLIT_FILES = {
    "train": PROCESSED_DIR / "train_clean.csv",
    "validation": PROCESSED_DIR / "validation_clean.csv",
    "test": PROCESSED_DIR / "test_clean.csv",
}
LABEL_NAMES = {0: "Fake", 1: "True"}
LABELS = [0, 1]
MAX_TEXT_LENGTH = 128
CPU_DEVICE = torch.device("cpu")


def _require_local_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required local artifact does not exist: {path}")


def _probability_record(probabilities: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Expected exactly two finite class probabilities")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Class probabilities must be between 0 and 1")
    if not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to 1")
    label = int(values.argmax())
    return {
        "predicted_label": label,
        "predicted_class": LABEL_NAMES[label],
        "probability_0_fake": float(values[0]),
        "probability_1_true": float(values[1]),
        "confidence": float(values[label]),
    }


def _load_tfidf_artifacts() -> tuple[Any, Any]:
    _require_local_file(TFIDF_PATH)
    _require_local_file(LOGISTIC_REGRESSION_PATH)
    import joblib

    vectorizer = joblib.load(TFIDF_PATH)
    classifier = joblib.load(LOGISTIC_REGRESSION_PATH)
    if list(classifier.classes_) != LABELS:
        raise ValueError("TF-IDF classifier must expose classes [0, 1]")
    return vectorizer, classifier


def _load_distilbert() -> tuple[Any, Any]:
    if not DISTILBERT_DIR.is_dir():
        raise FileNotFoundError(f"Local DistilBERT directory does not exist: {DISTILBERT_DIR}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(DISTILBERT_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(DISTILBERT_DIR), local_files_only=True
    ).to(CPU_DEVICE)
    model.eval()
    if model.config.num_labels != 2:
        raise ValueError("DistilBERT checkpoint must have two output labels")
    if model.config.id2label and {
        int(key): value for key, value in model.config.id2label.items()
    } != LABEL_NAMES:
        raise ValueError("DistilBERT checkpoint labels do not match 0=Fake, 1=True")
    return tokenizer, model


def _load_resnet() -> tuple[Any, Any]:
    _require_local_file(RESNET_CHECKPOINT_PATH)
    from torchvision import transforms
    from torchvision.models import resnet50

    checkpoint = torch.load(
        RESNET_CHECKPOINT_PATH, map_location=CPU_DEVICE, weights_only=True
    )
    if checkpoint.get("label_mapping") != {"0": "Fake", "1": "True"}:
        raise ValueError("ResNet checkpoint labels do not match 0=Fake, 1=True")
    if checkpoint.get("num_classes") != 2:
        raise ValueError("ResNet checkpoint must have two output classes")
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(CPU_DEVICE).eval()
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, image_transform


def _predict_distilbert(
    texts: Sequence[str], tokenizer: Any, model: Any, batch_size: int = 8
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**batch).logits
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(output, axis=0)


def _predict_resnet(
    image_paths: Sequence[Path],
    model: Any,
    image_transform: Any,
    batch_size: int = 16,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[start : start + batch_size]:
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {path}")
            try:
                with Image.open(path) as image:
                    images.append(image_transform(image.convert("RGB")))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                raise ValueError(f"Cannot read local image {path}: {exc}") from exc
        with torch.inference_mode():
            logits = model(torch.stack(images).to(CPU_DEVICE))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(output, axis=0)


def predict_one(text: str, image_path: Path | None = None) -> dict[str, Any]:
    """Run each saved text branch and, when supplied, the local ResNet branch."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    vectorizer, tfidf_model = _load_tfidf_artifacts()
    tfidf_probabilities = tfidf_model.predict_proba(vectorizer.transform([text]))[0]

    tokenizer, distilbert_model = _load_distilbert()
    distilbert_probabilities = _predict_distilbert(
        [text], tokenizer, distilbert_model
    )[0]

    predictions: dict[str, Any] = {
        "tfidf_logistic_regression": _probability_record(tfidf_probabilities),
        "distilbert": _probability_record(distilbert_probabilities),
    }
    if image_path is not None:
        model, transform = _load_resnet()
        probabilities = _predict_resnet([image_path], model, transform)[0]
        predictions["resnet50"] = _probability_record(probabilities)
    return {
        "label_mapping": {"0": "Fake", "1": "True"},
        "input": {
            "text": text,
            "local_image": str(image_path.resolve()) if image_path is not None else None,
        },
        "predictions": predictions,
    }


def _read_manifest() -> dict[str, Any]:
    _require_local_file(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        if split not in manifest:
            raise ValueError(f"Image manifest is missing split {split!r}")
        entry = manifest[split]
        ids = entry.get("verified_sample_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError(f"Manifest IDs for {split} are missing or duplicated")
        if entry.get("usable_images") != len(ids):
            raise ValueError(f"Manifest usable image count does not match IDs for {split}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(manifest[left]["verified_sample_ids"]) & set(
            manifest[right]["verified_sample_ids"]
        ):
            raise ValueError(f"Image manifest has ID overlap: {left} and {right}")
    return manifest


def _read_official_frame(split: str) -> pd.DataFrame:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown official split: {split}")
    _require_local_file(SPLIT_FILES[split])
    frame = pd.read_csv(SPLIT_FILES[split], usecols=["id", "text", "label"])
    if frame["id"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"Official {split} split has missing IDs or labels")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"Official {split} split has duplicate IDs")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin(LABELS).all():
        raise ValueError(f"Official {split} labels are not binary 0/1")
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def _aligned_subset(
    split: str, frame: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[list[str], np.ndarray, list[str], list[Path]]:
    entry = manifest[split]
    ids = [str(sample_id) for sample_id in entry["verified_sample_ids"]]
    indexed = frame.set_index("id", verify_integrity=True)
    missing_ids = sorted(set(ids) - set(indexed.index))
    if missing_ids:
        raise ValueError(f"Manifest {split} IDs absent from official CSV: {missing_ids[:5]}")
    subset = indexed.loc[ids]
    labels = subset["label"].to_numpy(dtype=np.int64)
    expected_counts = {
        str(label): int((labels == label).sum()) for label in LABELS
    }
    if expected_counts != entry.get("usable_label_counts"):
        raise ValueError(f"Manifest label counts do not match official {split} IDs")
    paths = [IMAGE_ROOT / split / f"{sample_id}.jpg" for sample_id in ids]
    missing_images = [str(path) for path in paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Manifest {split} image is missing (no substitutions allowed): "
            f"{missing_images[0]}"
        )
    return ids, labels, subset["text"].tolist(), paths


def _metrics(actual: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_binary_label_1": float(
            f1_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix_label_order_0_1": confusion_matrix(
            actual, predicted, labels=LABELS
        ).tolist(),
        "per_class": {
            f"{label}_{LABEL_NAMES[label]}": {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in LABELS
        },
    }


def _select_fusion_weight(
    actual: Sequence[int],
    text_probabilities: np.ndarray,
    image_probabilities: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Select text probability weight from validation binary F1 only."""
    candidates: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for hundredths in range(101):
        text_weight = hundredths / 100.0
        fused = (
            text_weight * text_probabilities[:, 1]
            + (1.0 - text_weight) * image_probabilities[:, 1]
        )
        predicted = (fused >= 0.5).astype(np.int64)
        metrics = _metrics(actual, predicted)
        tie_key = (
            metrics["f1_binary_label_1"],
            metrics["accuracy"],
            -abs(text_weight - 0.5),
            -text_weight,
        )
        candidates.append((tie_key, text_weight, metrics))
    _, selected_weight, selected_metrics = max(candidates, key=lambda item: item[0])
    return selected_weight, selected_metrics


def _fused_predictions(
    text_probabilities: np.ndarray,
    image_probabilities: np.ndarray,
    text_weight: float,
) -> np.ndarray:
    if text_probabilities.shape != image_probabilities.shape:
        raise ValueError("Text and image probabilities are not aligned")
    probability_true = (
        text_weight * text_probabilities[:, 1]
        + (1.0 - text_weight) * image_probabilities[:, 1]
    )
    return (probability_true >= 0.5).astype(np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance_hashes(include_test: bool) -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        MANIFEST_PATH,
        SPLIT_FILES["validation"],
        TFIDF_PATH,
        LOGISTIC_REGRESSION_PATH,
        RESNET_CHECKPOINT_PATH,
        DISTILBERT_DIR / "config.json",
        DISTILBERT_DIR / "model.safetensors",
        DISTILBERT_DIR / "tokenizer.json",
        DISTILBERT_DIR / "tokenizer_config.json",
        DISTILBERT_DIR / "vocab.txt",
    ]
    if include_test:
        paths.append(SPLIT_FILES["test"])
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
    }


def _new_result_path(requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else PROJECT_ROOT / path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiments" / f"multimodal_late_fusion_{stamp}.json"


def evaluate_fusion(output_path: str | None = None) -> dict[str, Any]:
    """Evaluate aligned image subsets; choose fusion weights on validation only."""
    started = time.perf_counter()
    output = _new_result_path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    manifest = _read_manifest()
    vectorizer, tfidf_model = _load_tfidf_artifacts()
    tokenizer, distilbert_model = _load_distilbert()
    resnet_model, image_transform = _load_resnet()

    validation_frame = _read_official_frame("validation")
    val_ids, val_actual, val_text, val_images = _aligned_subset(
        "validation", validation_frame, manifest
    )
    val_tfidf = tfidf_model.predict_proba(vectorizer.transform(val_text))
    val_distilbert = _predict_distilbert(val_text, tokenizer, distilbert_model)
    val_resnet = _predict_resnet(val_images, resnet_model, image_transform)
    val_tfidf_pred = val_tfidf.argmax(axis=1)
    val_distilbert_pred = val_distilbert.argmax(axis=1)
    val_resnet_pred = val_resnet.argmax(axis=1)

    tfidf_weight, val_tfidf_fusion_metrics = _select_fusion_weight(
        val_actual, val_tfidf, val_resnet
    )
    distilbert_weight, val_distilbert_fusion_metrics = _select_fusion_weight(
        val_actual, val_distilbert, val_resnet
    )

    # Test data is not read or inferred until both fusion weights are fixed.
    test_frame = _read_official_frame("test")
    test_ids, test_actual, test_text, test_images = _aligned_subset(
        "test", test_frame, manifest
    )
    test_tfidf = tfidf_model.predict_proba(vectorizer.transform(test_text))
    test_distilbert = _predict_distilbert(test_text, tokenizer, distilbert_model)
    test_resnet = _predict_resnet(test_images, resnet_model, image_transform)

    results = {
        "experiment": "Exploratory validation-selected late fusion on manifest-aligned image subsets",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_mapping": {"0": "Fake", "1": "True"},
        "scope": {
            "official_processed_split_rows": {
                "validation": len(validation_frame),
                "test": len(test_frame),
            },
            "evaluated_manifest_subset_rows": {
                "validation": len(val_ids),
                "test": len(test_ids),
            },
            "full_split_evaluation": False,
            "subset_only": True,
            "official_image_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "selection_rule": (
                "For each text branch, select alpha on validation from 0.00 to 1.00 "
                "in steps of 0.01 maximizing binary F1 for label 1; ties resolve by "
                "higher accuracy, then closeness to 0.50, then lower alpha."
            ),
            "fusion_equation": (
                "p_true_fused = alpha * p_true_text + (1 - alpha) * p_true_resnet; "
                "predict 1 when p_true_fused >= 0.5"
            ),
            "test_selection_or_tuning": False,
        },
        "data": {
            "validation": {
                "official_rows_available": len(validation_frame),
                "manifest_ids_evaluated": len(val_ids),
                "official_rows_outside_subset": len(validation_frame) - len(val_ids),
                "failed_or_unreadable_manifest_candidates": manifest["validation"].get(
                    "failed_or_unreadable_attempts"
                ),
                "label_counts": manifest["validation"]["usable_label_counts"],
            },
            "test": {
                "official_rows_available": len(test_frame),
                "manifest_ids_evaluated": len(test_ids),
                "official_rows_outside_subset": len(test_frame) - len(test_ids),
                "failed_or_unreadable_manifest_candidates": manifest["test"].get(
                    "failed_or_unreadable_attempts"
                ),
                "label_counts": manifest["test"]["usable_label_counts"],
            },
            "manifest_validation_ids": val_ids,
            "manifest_test_ids": test_ids,
            "images_downloaded_or_substituted": False,
        },
        "errors": [
            {
                "attempt": 1,
                "phase": "initial local ResNet checkpoint load",
                "error": (
                    "Checkpoint FC weights have shape [2, 2048], while the "
                    "unmodified torchvision model FC layer has shape [1000, 2048]."
                ),
                "resolution": "Replaced the default 1000-class FC layer with the checkpoint's two-class head.",
                "official_validation_or_test_rows_read": False,
            }
        ],
        "configuration": {
            "device": "cpu",
            "seed": None,
            "prediction_batch_size": {
                "distilbert": 8,
                "resnet50": 16,
            },
            "distilbert_max_length": MAX_TEXT_LENGTH,
            "resnet_preprocessing": {
                "resize_shorter_side": 256,
                "center_crop": 224,
                "normalization": "ImageNet mean/std [0.485,0.456,0.406]/[0.229,0.224,0.225]",
            },
            "fused_probability_threshold": 0.5,
            "fusion_alpha_grid_step": 0.01,
        },
        "models": {
            "tfidf_logistic_regression": {
                "input": "clean_title/text only",
                "probabilities": "saved TF-IDF vectorizer and LogisticRegression predict_proba",
            },
            "distilbert": {
                "input": "clean_title/text only",
                "probabilities": "local saved DistilBERT softmax",
            },
            "resnet50": {
                "input": "manifest-listed local image only",
                "checkpoint": str(RESNET_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            },
        },
        "validation": {
            "individual_models": {
                "tfidf_logistic_regression": _metrics(
                    val_actual, val_tfidf_pred
                ),
                "distilbert": _metrics(val_actual, val_distilbert_pred),
                "resnet50": _metrics(val_actual, val_resnet_pred),
            },
            "selected_fusions": {
                "tfidf_plus_resnet50": {
                    "text_probability_weight_alpha": tfidf_weight,
                    "metrics": val_tfidf_fusion_metrics,
                },
                "distilbert_plus_resnet50": {
                    "text_probability_weight_alpha": distilbert_weight,
                    "metrics": val_distilbert_fusion_metrics,
                },
            },
        },
        "test": {
            "individual_models": {
                "tfidf_logistic_regression": _metrics(
                    test_actual, test_tfidf.argmax(axis=1)
                ),
                "distilbert": _metrics(
                    test_actual, test_distilbert.argmax(axis=1)
                ),
                "resnet50": _metrics(test_actual, test_resnet.argmax(axis=1)),
            },
            "selected_fusions": {
                "tfidf_plus_resnet50": {
                    "text_probability_weight_alpha": tfidf_weight,
                    "metrics": _metrics(
                        test_actual,
                        _fused_predictions(test_tfidf, test_resnet, tfidf_weight),
                    ),
                },
                "distilbert_plus_resnet50": {
                    "text_probability_weight_alpha": distilbert_weight,
                    "metrics": _metrics(
                        test_actual,
                        _fused_predictions(
                            test_distilbert, test_resnet, distilbert_weight
                        ),
                    ),
                },
            },
        },
        "environment": {
            "sys_executable": sys.executable,
            "sys_version": sys.version,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "duration_seconds": time.perf_counter() - started,
        "provenance_sha256": _provenance_hashes(include_test=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never replace a prior experiment result.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"New result saved: {output}")
    print(
        json.dumps(
            {
                "validation": results["validation"],
                "test": results["test"],
                "duration_seconds": results["duration_seconds"],
                "result_path": str(output),
            },
            indent=2,
        )
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Fakeddit text and optional image inference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser(
        "predict", help="Predict from text and an optional local image"
    )
    predict_parser.add_argument("--text", required=True, help="Input text")
    predict_parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image path; no URL downloads are performed",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate-fusion",
        help="Select late-fusion weights on aligned validation IDs, then test once",
    )
    evaluate_parser.add_argument(
        "--output",
        help="New result JSON path (default: timestamped file under experiments/)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "predict":
        if args.image is not None and not args.image.is_file():
            raise FileNotFoundError(f"Local image path does not exist: {args.image}")
        print(json.dumps(predict_one(args.text, args.image), indent=2))
        return 0
    if args.command == "evaluate-fusion":
        evaluate_fusion(args.output)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
