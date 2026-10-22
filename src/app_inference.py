"""Small inference helpers shared by the Streamlit interface and tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Text input must be a string.")
    normalized = text.strip()
    if not normalized:
        raise ValueError("Enter a headline or text before analysis.")
    return normalized


def validate_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Image input must be a decoded PIL image.")
    if image.width < 1 or image.height < 1:
        raise ValueError("The uploaded image is empty.")
    if image.width * image.height > 40_000_000:
        raise ValueError("The image is too large to process (maximum 40 megapixels).")
    return image.convert("RGB")


def predict_resnet_image(
    image: Image.Image, model: Any, image_transform: Any
) -> np.ndarray:
    """Return local ResNet class probabilities for one validated image."""
    rgb_image = validate_image(image)
    tensor = image_transform(rgb_image).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    if probabilities.shape != (1, 2):
        raise ValueError(
            f"ResNet returned an unexpected probability shape: {probabilities.shape}"
        )
    return probabilities[0]


def late_fusion_probabilities(
    text_probabilities: Any,
    image_probabilities: Any,
    text_weight: float,
) -> np.ndarray:
    """Combine text/image class probabilities using the recorded alpha rule."""
    text = np.asarray(text_probabilities, dtype=np.float64)
    image = np.asarray(image_probabilities, dtype=np.float64)
    if text.shape != (2,) or image.shape != (2,):
        raise ValueError("Fusion requires two class probabilities from each model.")
    for name, values in (("text", text), ("image", image)):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name.title()} probabilities must be finite values in [0, 1].")
        if not np.isclose(values.sum(), 1.0, atol=1e-5):
            raise ValueError(f"{name.title()} class probabilities must sum to 1.")
    if not np.isfinite(text_weight) or not 0.0 <= text_weight <= 1.0:
        raise ValueError("Fusion text weight must be between 0 and 1.")
    true_probability = (
        text_weight * text[1] + (1.0 - text_weight) * image[1]
    )
    return np.asarray([1.0 - true_probability, true_probability], dtype=np.float64)


def probability_record(probabilities: Any) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Expected exactly two finite class probabilities.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Class probabilities must be between 0 and 1.")
    if not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to 1.")
    label = int(values.argmax())
    return {
        "predicted_label": label,
        "predicted_class": "Fake" if label == 0 else "True",
        "probability_0_fake": float(values[0]),
        "probability_1_true": float(values[1]),
        "confidence": float(values[label]),
    }
