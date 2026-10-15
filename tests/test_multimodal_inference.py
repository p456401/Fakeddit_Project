"""Focused tests for the local multimodal inference and fusion utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import multimodal_inference as inference  # noqa: E402


class ProbabilityRecordTests(unittest.TestCase):
    def test_probability_record_uses_project_label_mapping(self) -> None:
        result = inference._probability_record([0.2, 0.8])
        self.assertEqual(result["predicted_label"], 1)
        self.assertEqual(result["predicted_class"], "True")
        self.assertAlmostEqual(result["probability_0_fake"], 0.2)
        self.assertAlmostEqual(result["probability_1_true"], 0.8)
        self.assertAlmostEqual(result["confidence"], 0.8)

    def test_probability_record_rejects_non_normalized_values(self) -> None:
        with self.assertRaises(ValueError):
            inference._probability_record([0.4, 0.4])


class FusionTests(unittest.TestCase):
    def test_fused_prediction_combines_positive_class_probabilities(self) -> None:
        text = np.asarray([[0.1, 0.9], [0.7, 0.3]])
        image = np.asarray([[0.8, 0.2], [0.2, 0.8]])
        prediction = inference._fused_predictions(text, image, 0.75)
        np.testing.assert_array_equal(prediction, [1, 0])

    def test_weight_selection_uses_validation_binary_f1(self) -> None:
        actual = np.asarray([1, 1, 0, 0])
        text = np.asarray([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
        image = np.asarray([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        alpha, selected_metrics = inference._select_fusion_weight(
            actual, text, image
        )
        self.assertGreater(alpha, 0.5)
        self.assertEqual(selected_metrics["f1_binary_label_1"], 1.0)

    def test_metric_report_has_per_class_fake_and_true(self) -> None:
        result = inference._metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(
            set(result["per_class"]), {"0_Fake", "1_True"}
        )
        self.assertEqual(
            result["confusion_matrix_label_order_0_1"], [[1, 1], [1, 1]]
        )


class ResultSafetyTests(unittest.TestCase):
    def test_explicit_relative_output_resolves_under_project_root(self) -> None:
        result = inference._new_result_path("experiments/new-result.json")
        self.assertEqual(
            result, inference.PROJECT_ROOT / "experiments" / "new-result.json"
        )

    def test_cli_help_is_registered(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            inference._build_parser().parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
