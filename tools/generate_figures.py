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
    ("TF-IDF + Logistic Regression\n59,163 full-test rows", PROJECT_ROOT / "experiments" / "baseline_results.json", None),
    ("DistilBERT\n59,163 full-test rows", PROJECT_ROOT / "experiments" / "transformer_results.json", None),
    ("ResNet-50\n150 test images", PROJECT_ROOT / "experiments" / "resnet_results.json", None),
    ("TF-IDF + ResNet-50\n150 aligned test examples", FUSION_RESULTS, "tfidf_plus_resnet50"),
    ("DistilBERT + ResNet-50\n150 aligned test examples", FUSION_RESULTS, "distilbert_plus_resnet50"),
)
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision (True)",
    "recall": "Recall (True)",
    "f1": "F1 (True)",
    "macro_f1": "Macro F1",
}


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
    if model.startswith("TF-IDF") and isinstance(metrics, dict):
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
    fig, ax = plt.subplots(figsize=(11.3, 5.5), layout="constrained")
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
            label=METRIC_LABELS[metric],
            color=colors[metric_index],
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.0%}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_ylim(0, 1.20)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2), [f"{value:.0%}" for value in np.arange(0.0, 1.01, 0.2)])
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_title(
        "Held-out test metrics by evaluation scope\n"
        "Text models: full official test split | Image and fusion: 150-example subset",
        fontsize=12,
        pad=18,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.axvline(1.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(0.5, 1.125, "FULL OFFICIAL TEST SPLIT", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#365F91")
    ax.text(3.0, 1.125, "ALIGNED IMAGE SUBSET", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#9C5700")
    ax.text(0.5, -0.30,
            "Positive class: True (label 1). Macro F1 averages labels 0 and 1. "
            "TF-IDF full-test Macro F1 was not recorded.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    ax.legend(
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=9,
    )
    fig.savefig(args.output_dir / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    full_test_indices = (0, 1)
    subset_indices = (2, 3, 4)
    for filename, selected_indices, scope_title in (
        ("confusion_matrices_full_test.png", full_test_indices, "Full official test split: 59,163 rows per model"),
        ("confusion_matrices_subset.png", subset_indices, "Held-out image/fusion subset: 150 aligned examples per model"),
    ):
        selected_names = [labels[index] for index in selected_indices]
        selected_metrics = [loaded_metrics[index] for index in selected_indices]
        figure_width = 10.5 if len(selected_indices) == 2 else 11.3
        fig, axes = plt.subplots(
            1,
            len(selected_indices),
            figsize=(figure_width, 5.2),
            layout="constrained",
            squeeze=False,
        )
        vmax = max(
            int(np.asarray(metric.get("confusion_matrix"), dtype=int).max())
            for metric in selected_metrics
        )
        images = []
        for ax, name, metrics in zip(axes[0], selected_names, selected_metrics):
            matrix = np.asarray(metrics.get("confusion_matrix"), dtype=int)
            if matrix.shape != (2, 2):
                raise ValueError(f"{name} confusion matrix must be 2x2")
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            images.append(image)
            ax.set_title(name, fontsize=10, pad=12)
            ax.set_xticks((0, 1), ("Fake (0)", "True (1)"))
            ax.set_yticks((0, 1), ("Fake (0)", "True (1)"))
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.tick_params(axis="both", labelsize=9)
            for row in range(2):
                for column in range(2):
                    ax.text(
                        column,
                        row,
                        f"{matrix[row, column]:,}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white" if matrix[row, column] > vmax / 2 else "black",
                    )
        fig.suptitle(
            f"Test confusion matrices — {scope_title}\n"
            "Rows: actual class | Columns: predicted class | Order: Fake (0), True (1)",
            fontsize=11,
        )
        fig.colorbar(images[0], ax=list(axes[0]), shrink=0.82, label="Number of examples")
        fig.savefig(args.output_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(f"Generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
