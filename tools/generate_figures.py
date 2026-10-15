"""Render report figures from the saved experiment JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_RESULTS = (
    PROJECT_ROOT / "experiments" / "multimodal_late_fusion_20261015_140619.json"
)
RESULTS = (
    ("TF-IDF\n(full test)", PROJECT_ROOT / "experiments" / "baseline_results.json", None),
    ("DistilBERT\n(full test)", PROJECT_ROOT / "experiments" / "transformer_results.json", None),
    ("ResNet-50\n(150)", PROJECT_ROOT / "experiments" / "resnet_results.json", None),
    ("TF-IDF + ResNet\n(150 aligned)", FUSION_RESULTS, "tfidf_plus_resnet50"),
    ("DistilBERT + ResNet\n(150 aligned)", FUSION_RESULTS, "distilbert_plus_resnet50"),
)
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1")


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def test_metrics(
    result: dict[str, Any], model: str, fusion_key: str | None
) -> dict[str, Any]:
    if fusion_key is not None:
        metrics = result["test"]["selected_fusions"][fusion_key]["metrics"]
        positive_class = metrics["per_class"]["1_True"]
        return {
            **metrics,
            "precision": positive_class["precision"],
            "recall": positive_class["recall"],
            "f1": metrics["f1_binary_label_1"],
            "macro_f1": metrics["f1_macro"],
            "confusion_matrix": metrics["confusion_matrix_label_order_0_1"],
        }
    metrics = result.get("test_metrics")
    if model == "TF-IDF" and isinstance(metrics, dict):
        metrics = {
            **metrics,
            "f1": metrics.get("f1", metrics.get("f1_score")),
            "confusion_matrix": result.get("test_confusion_matrix"),
        }
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing test metrics for {model}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures",
        help="Directory for generated PNG figures.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    loaded_metrics: list[dict[str, Any]] = []
    for name, path, fusion_key in RESULTS:
        result = load_result(path)
        labels.append(name)
        loaded_metrics.append(test_metrics(result, name.split("\n")[0], fusion_key))

    colors = ("#4472C4", "#70AD47", "#ED7D31", "#A5A5A5", "#8064A2")
    fig, ax = plt.subplots(figsize=(13, 6.2), layout="constrained")
    x = np.arange(len(labels))
    metrics_available = [
        (metric, index)
        for index, metric in enumerate(METRICS)
        if any(isinstance(row.get(metric), (int, float)) for row in loaded_metrics)
    ]
    width = 0.8 / max(len(metrics_available), 1)
    for slot, (metric, metric_index) in enumerate(metrics_available):
        positions = x - 0.4 + width * (slot + 0.5)
        values = [
            float(row[metric]) if isinstance(row.get(metric), (int, float)) else np.nan
            for row in loaded_metrics
        ]
        bars = ax.bar(
            positions,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[metric_index],
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_title("Saved test metrics (different test scopes; not a controlled ranking)")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(labels), figsize=(18, 4.7), layout="constrained"
    )
    for ax, name, metrics in zip(axes, labels, loaded_metrics):
        matrix = np.asarray(metrics.get("confusion_matrix"), dtype=int)
        if matrix.shape != (2, 2):
            raise ValueError(f"{name} confusion matrix must be 2x2")
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks((0, 1), ("Fake (0)", "True (1)"), rotation=25, ha="right")
        ax.set_yticks((0, 1), ("Fake (0)", "True (1)"))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Saved test confusion matrices (actual rows, predicted columns)")
    fig.savefig(args.output_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    print(f"Generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
