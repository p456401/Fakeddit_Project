from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import app_inference  # noqa: E402


class InputValidationTests(unittest.TestCase):
    def test_text_is_trimmed(self) -> None:
        self.assertEqual(app_inference.validate_text("  example claim  "), "example claim")

    def test_blank_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter a headline"):
            app_inference.validate_text("   ")

    def test_valid_image_is_converted_to_rgb(self) -> None:
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        result = app_inference.validate_image(image)
        self.assertEqual(result.mode, "RGB")

    def test_oversized_image_is_rejected(self) -> None:
        image = Image.Image()
        image._size = (7000, 6000)
        with self.assertRaisesRegex(ValueError, "40 megapixels"):
            app_inference.validate_image(image)


class FusionTests(unittest.TestCase):
    def test_late_fusion_uses_saved_probability_weight_rule(self) -> None:
        result = app_inference.late_fusion_probabilities(
            [0.1, 0.9], [0.8, 0.2], 0.63
        )
        np.testing.assert_allclose(result, [0.359, 0.641])
        self.assertEqual(int(result.argmax()), 1)

    def test_fusion_rejects_malformed_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            app_inference.late_fusion_probabilities([0.3, 0.3], [0.2, 0.8], 0.5)

    def test_prediction_record_preserves_fake_true_mapping(self) -> None:
        result = app_inference.probability_record([0.8, 0.2])
        self.assertEqual(result["predicted_label"], 0)
        self.assertEqual(result["predicted_class"], "Fake")
        self.assertAlmostEqual(result["probability_1_true"], 0.2)

    def test_resnet_predictor_runs_on_a_small_local_image(self) -> None:
        class TinyModel(torch.nn.Module):
            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                mean = tensor.mean(dim=(1, 2, 3))
                return torch.stack((1.0 - mean, mean), dim=1)

        transform = lambda image: torch.from_numpy(  # noqa: E731
            np.asarray(image.resize((2, 2)), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        image = Image.new("RGB", (8, 8), (255, 255, 255))
        result = app_inference.predict_resnet_image(image, TinyModel(), transform)
        np.testing.assert_allclose(result, [0.26894142, 0.73105858])


if __name__ == "__main__":
    unittest.main()
