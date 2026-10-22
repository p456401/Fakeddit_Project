"""Streamlit interface for the saved Fakeddit misinformation classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import app_inference as app_models  # noqa: E402
import multimodal_inference as inference  # noqa: E402


RESULTS = {
    "TF-IDF + Logistic Regression": ROOT / "experiments" / "baseline_results.json",
    "DistilBERT": ROOT / "experiments" / "transformer_results.json",
    "ResNet-50": ROOT / "experiments" / "resnet_results.json",
}
FUSION_RESULTS = (
    ROOT / "experiments" / "multimodal_late_fusion_20261015_140619.json"
)
DISPLAY_NAMES = {
    "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
    "distilbert": "DistilBERT",
    "resnet50": "ResNet-50",
    "tfidf_plus_resnet50": "TF-IDF + ResNet-50 late fusion",
    "distilbert_plus_resnet50": "DistilBERT + ResNet-50 late fusion",
}

st.set_page_config(
    page_title="Fakeddit | Misinformation Verification",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101b31; }
    [data-testid="stSidebar"] * { color: #edf3ff; }
    .hero {
      padding: 2.2rem 2.4rem; border-radius: 20px;
      background: linear-gradient(115deg, #10213d, #1b5270 66%, #287d7d);
      color: white; margin: .5rem 0 1.5rem;
    }
    .hero h1 { color: white; font-size: 2.4rem; margin: 0 0 .55rem; }
    .hero p { color: #e4efff; max-width: 850px; font-size: 1.05rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em;
      color: #8bd5cb; font-weight: 700; font-size: .78rem; }
    .model-card {
      border: 1px solid #dce5ef; background: white; border-radius: 14px;
      padding: 1.15rem 1.3rem; min-height: 185px;
      box-shadow: 0 4px 18px rgba(16, 33, 61, .05);
    }
    .muted { color: #5c6b7c; }
    .notice {
      border-left: 4px solid #dfaa36; padding: .85rem 1rem;
      background: #fff8e6; border-radius: 6px; color: #493c22;
    }
    div[data-testid="stMetric"] {
      background: white; border: 1px solid #dce5ef; border-radius: 12px;
      padding: .8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_tfidf() -> tuple[Any, Any]:
    return inference._load_tfidf_artifacts()


@st.cache_resource(show_spinner=False)
def load_distilbert() -> tuple[Any, Any]:
    return inference._load_distilbert()


@st.cache_resource(show_spinner=False)
def load_resnet() -> tuple[Any, Any]:
    return inference._load_resnet()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Fakeddit · Machine-learning project</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Saved experiment record was not found: {path.name}")
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Experiment record is not a JSON object: {path.name}")
    return result


def format_metric(value: Any) -> str:
    if not isinstance(value, (float, int)):
        return "Not recorded"
    return f"{float(value):.2%}"


def show_prediction_card(name: str, prediction: dict[str, Any] | None, error: Exception | None) -> None:
    if error is not None:
        st.error(f"**{name} unavailable.** {error}")
        return
    if prediction is None:
        st.info(f"**{name}:** not run for this input.")
        return
    label = prediction["predicted_class"]
    st.markdown(
        f"""
        <div class="model-card">
          <div class="eyebrow" style="color:#287d7d">{name}</div>
          <h3 style="margin:.5rem 0">Prediction: {label}
            <span class="muted">({prediction['predicted_label']})</span></h3>
          <div>Fake (0) score: <strong>{prediction['probability_0_fake']:.1%}</strong></div>
          <div>True (1) score: <strong>{prediction['probability_1_true']:.1%}</strong></div>
          <div class="muted" style="margin-top:.55rem">
            Predicted-class score: {prediction['confidence']:.1%}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(text: str, image: Image.Image | None) -> dict[str, Any]:
    result: dict[str, Any] = {"predictions": {}, "errors": {}, "fusion": {}}
    tfidf_probs = None
    bert_probs = None
    image_probs = None
    if text:
        try:
            vectorizer, classifier = load_tfidf()
            tfidf_probs = classifier.predict_proba(vectorizer.transform([text]))[0]
            result["predictions"]["tfidf_logistic_regression"] = (
                app_models.probability_record(tfidf_probs)
            )
        except Exception as exc:
            result["errors"]["tfidf_logistic_regression"] = exc
        try:
            tokenizer, model = load_distilbert()
            bert_probs = inference._predict_distilbert(
                [text], tokenizer, model
            )[0]
            result["predictions"]["distilbert"] = app_models.probability_record(
                bert_probs
            )
        except Exception as exc:
            result["errors"]["distilbert"] = exc
    if image is not None:
        try:
            image_model, image_transform = load_resnet()
            image_probs = app_models.predict_resnet_image(
                image, image_model, image_transform
            )
            result["predictions"]["resnet50"] = app_models.probability_record(
                image_probs
            )
        except Exception as exc:
            result["errors"]["resnet50"] = exc
    if image_probs is not None and (tfidf_probs is not None or bert_probs is not None):
        try:
            saved_fusion = read_json(FUSION_RESULTS)
            selected = saved_fusion["test"]["selected_fusions"]
            if tfidf_probs is not None:
                alpha = float(
                    selected["tfidf_plus_resnet50"][
                        "text_probability_weight_alpha"
                    ]
                )
                probabilities = app_models.late_fusion_probabilities(
                    tfidf_probs, image_probs, alpha
                )
                result["fusion"]["tfidf_plus_resnet50"] = {
                    "prediction": app_models.probability_record(probabilities),
                    "alpha": alpha,
                }
            if bert_probs is not None:
                alpha = float(
                    selected["distilbert_plus_resnet50"][
                        "text_probability_weight_alpha"
                    ]
                )
                probabilities = app_models.late_fusion_probabilities(
                    bert_probs, image_probs, alpha
                )
                result["fusion"]["distilbert_plus_resnet50"] = {
                    "prediction": app_models.probability_record(probabilities),
                    "alpha": alpha,
                }
        except Exception as exc:
            result["errors"]["fusion"] = exc
    return result


def render_home() -> None:
    hero(
        "Context-aware multimodal misinformation verification",
        "An academic demonstration of text and image classification on Fakeddit. "
        "Each model reports its own output; the application does not establish "
        "whether a real-world claim is objectively true.",
    )
    st.markdown("### A transparent, model-by-model workflow")
    columns = st.columns(3)
    for column, number, title, description in zip(
        columns,
        ("01", "02", "03"),
        ("Provide evidence", "Inspect model outputs", "Review evaluation"),
        (
            "Enter a headline or text. A local image is optional.",
            "Compare independent text/image classifiers and, when both modalities are present, the recorded late-fusion rule.",
            "Explore saved validation/test results with sample sizes and limitations shown.",
        ),
    ):
        with column:
            st.markdown(
                f'<div class="model-card"><div class="eyebrow">{number}</div>'
                f"<h3>{title}</h3><p class='muted'>{description}</p></div>",
                unsafe_allow_html=True,
            )
    st.markdown("")
    first, second, third = st.columns(3)
    first.metric("Saved text test scope", "59,163 rows")
    second.metric("ResNet / fusion test scope", "150 images")
    third.metric("Label convention", "0 Fake · 1 True")
    st.markdown(
        '<div class="notice"><strong>Research status:</strong> late fusion is '
        "exploratory and uses 150 aligned validation and test examples. CLIP "
        "consistency and context-metadata modeling remain incomplete.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Outputs are dataset-trained classification scores, not calibrated "
        "fact-checking probabilities or a substitute for source verification."
    )


def render_analysis() -> None:
    hero(
        "Analyze a claim",
        "Run the saved local classifiers on text and an optional image. "
        "Model weights are loaded on demand and cached for this session.",
    )
    with st.form("analysis_form", clear_on_submit=False):
        text = st.text_area(
            "Headline or claim text",
            height=145,
            placeholder="Paste the text you want the models to classify...",
            help="TF-IDF and DistilBERT require text; ResNet can run on an image alone.",
        )
        uploaded = st.file_uploader(
            "Optional local image",
            type=("jpg", "jpeg", "png", "webp", "bmp"),
            help="The image is processed locally and is not sent to an external service.",
        )
        submitted = st.form_submit_button(
            "Analyze with available models", type="primary", width="stretch"
        )

    image: Image.Image | None = None
    image_error: str | None = None
    if uploaded is not None:
        try:
            opened_image = Image.open(uploaded)
            opened_image.verify()
            uploaded.seek(0)
            opened_image = Image.open(uploaded)
            image = ImageOps.exif_transpose(
                app_models.validate_image(opened_image)
            )
            st.image(image, caption="Image supplied for local ResNet inference", width=280)
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            image_error = f"Image could not be validated: {exc}"
            image = None
    if submitted:
        normalized_text = text.strip()
        if image_error:
            st.error(image_error)
        if not normalized_text and image is None:
            st.warning("Enter text or upload a valid local image to begin.")
            return
        if normalized_text:
            try:
                normalized_text = app_models.validate_text(normalized_text)
            except ValueError as exc:
                st.warning(str(exc))
                return
        with st.spinner("Loading available local models and analyzing input…"):
            outcome = run_analysis(normalized_text, image)
        st.session_state["last_analysis"] = {
            "input_text": normalized_text,
            "has_image": image is not None,
            "outcome": outcome,
        }

    analysis = st.session_state.get("last_analysis")
    if not analysis:
        st.info("Your results will appear here after you run an analysis.")
        return
    outcome = analysis["outcome"]
    st.markdown("### Individual model outputs")
    columns = st.columns(3)
    model_keys = (
        "tfidf_logistic_regression",
        "distilbert",
        "resnet50",
    )
    for column, key in zip(columns, model_keys):
        with column:
            show_prediction_card(
                DISPLAY_NAMES[key],
                outcome["predictions"].get(key),
                outcome["errors"].get(key),
            )
    st.caption(
        "The displayed class scores come from each model's output distribution. "
        "They are not calibrated probabilities that the underlying claim is factual."
    )

    st.markdown("### Text + image late fusion")
    if not outcome["fusion"]:
        st.info(
            "Late fusion requires both a successful text prediction and a "
            "successful local-image prediction. CLIP image-text consistency "
            "is not available in this project build."
        )
    else:
        st.warning(
            "Fusion uses weights selected on a small 150-example validation "
            "subset. Treat these outputs as an exploratory demonstration, not "
            "a general-purpose multimodal verification score."
        )
        columns = st.columns(len(outcome["fusion"]))
        for column, (key, record) in zip(columns, outcome["fusion"].items()):
            with column:
                st.markdown(
                    f"**{DISPLAY_NAMES[key]}** · validation-selected "
                    f"text weight α={record['alpha']:.2f}"
                )
                show_prediction_card(
                    DISPLAY_NAMES[key],
                    record["prediction"],
                    None,
                )
    if "fusion" in outcome["errors"]:
        st.error(f"Late fusion unavailable: {outcome['errors']['fusion']}")


def _result_rows() -> list[dict[str, Any]]:
    rows = []
    for model, path in RESULTS.items():
        record = read_json(path)
        metrics = record.get("test_metrics", {})
        if model == "TF-IDF + Logistic Regression":
            scope = "59,163 official test rows"
            f1 = metrics.get("f1_score")
            macro_f1 = None
            metrics["confusion_matrix"] = record.get("test_confusion_matrix")
        elif model == "DistilBERT":
            scope = f"{record.get('test_rows_evaluated', 59163):,} official test rows"
            f1 = metrics.get("f1")
            macro_f1 = metrics.get("macro_f1")
        else:
            scope = "150 held-out test images"
            f1 = metrics.get("f1")
            macro_f1 = metrics.get("macro_f1")
        rows.append(
            {
                "Model": model,
                "Accuracy": metrics.get("accuracy"),
                "Precision (True)": metrics.get("precision"),
                "Recall (True)": metrics.get("recall"),
                "F1 (True)": f1,
                "Macro F1": macro_f1,
                "Test scope": scope,
                "_metrics": metrics,
                "_path": path,
            }
        )
    fusion = read_json(FUSION_RESULTS)
    for key in ("tfidf_plus_resnet50", "distilbert_plus_resnet50"):
        item = fusion["test"]["selected_fusions"][key]
        metrics = item["metrics"]
        true_class = metrics["per_class"]["1_True"]
        rows.append(
            {
                "Model": DISPLAY_NAMES[key],
                "Accuracy": metrics["accuracy"],
                "Precision (True)": true_class["precision"],
                "Recall (True)": true_class["recall"],
                "F1 (True)": metrics["f1_binary_label_1"],
                "Macro F1": metrics["f1_macro"],
                "Test scope": "150 aligned held-out examples",
                "_metrics": metrics,
                "_path": FUSION_RESULTS,
            }
        )
    return rows


def render_results() -> None:
    hero(
        "Saved evaluation results",
        "Metrics are read directly from the experiment JSON files. Full-split "
        "text evaluations and small image/fusion subsets are deliberately distinguished.",
    )
    try:
        rows = _result_rows()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        st.error(f"Could not read saved experiment records: {exc}")
        return
    display = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    )
    for metric in (
        "Accuracy",
        "Precision (True)",
        "Recall (True)",
        "F1 (True)",
        "Macro F1",
    ):
        display[metric] = display[metric].map(format_metric)
    st.dataframe(display, hide_index=True, width="stretch")
    chart_data = pd.DataFrame(
        [
            {
                "Model": row["Model"],
                "Accuracy": row["Accuracy"],
                "Precision (True)": row["Precision (True)"],
                "Recall (True)": row["Recall (True)"],
                "F1 (True)": row["F1 (True)"],
                "Macro F1": row["Macro F1"],
            }
            for row in rows
        ]
    ).set_index("Model")
    st.markdown("### Metric comparison")
    st.bar_chart(
        chart_data,
        y_label="Saved score",
        x_label="Model (test scopes differ)",
        height=340,
    )
    st.caption(
        "Binary Precision, Recall and F1 use positive label 1 (True). Macro F1 "
        "averages labels 0 and 1. TF-IDF full-test Macro F1 was not recorded."
    )
    st.warning(
        "The TF-IDF and DistilBERT saved results use all 59,163 official test "
        "rows. ResNet and late fusion use a 150-example test subset; do not "
        "interpret cross-scope differences as a controlled ranking."
    )
    selected = st.selectbox("Confusion matrix", [row["Model"] for row in rows])
    row = next(item for item in rows if item["Model"] == selected)
    metrics = row["_metrics"]
    matrix = metrics.get("confusion_matrix")
    if matrix is None:
        matrix = metrics.get("confusion_matrix_label_order_0_1")
    if matrix is not None:
        array = np.asarray(matrix, dtype=int)
        if array.shape != (2, 2):
            st.error("Saved confusion matrix is not 2×2; it cannot be displayed.")
        else:
            fig, ax = plt.subplots(figsize=(5.1, 4.1), layout="constrained")
            image = ax.imshow(array, cmap="Blues")
            ax.set_xticks((0, 1), ("Fake (0)", "True (1)"))
            ax.set_yticks((0, 1), ("Fake (0)", "True (1)"))
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"{selected}\n{row['Test scope']}")
            for actual in range(2):
                for predicted in range(2):
                    ax.text(
                        predicted,
                        actual,
                        f"{array[actual, predicted]:,}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if array[actual, predicted] > array.max() / 2
                            else "#10213d"
                        ),
                    )
            fig.colorbar(image, ax=ax, label="Examples")
            st.pyplot(fig, width="content")
            plt.close(fig)
    if row["Test scope"] == "150 aligned held-out examples":
        st.caption(
            "Fusion weights were chosen using validation only. This test "
            "measurement includes 150 aligned examples from the official split."
        )


def render_methodology() -> None:
    hero(
        "Methodology",
        "Three independently trained branches plus an explicitly exploratory "
        "probability late-fusion baseline.",
    )
    columns = st.columns(3)
    methods = (
        (
            "TF-IDF + Logistic Regression",
            "Unigrams and bigrams; 50,000-feature cap; sublinear term frequency; "
            "balanced Logistic Regression.",
            "Text only",
        ),
        (
            "DistilBERT",
            "distilbert-base-uncased; 8,000 train and 8,000 validation rows; "
            "two epochs; class-weighted loss; CPU.",
            "Text only",
        ),
        (
            "ResNet-50",
            "ImageNet V2 initialization; frozen backbone; learned binary head; "
            "500 train, 150 validation and 150 test images.",
            "Image only",
        ),
    )
    for column, (title, description, modality) in zip(columns, methods):
        with column:
            st.markdown(
                f'<div class="model-card"><div class="eyebrow">{modality}</div>'
                f"<h3>{title}</h3><p class='muted'>{description}</p></div>",
                unsafe_allow_html=True,
            )
    st.markdown("### Late-fusion rule")
    st.latex(
        r"p_{\mathrm{True}}^{\mathrm{fused}}="
        r"\alpha p_{\mathrm{True}}^{\mathrm{text}}"
        r"+(1-\alpha)p_{\mathrm{True}}^{\mathrm{ResNet}}"
    )
    st.write(
        "Separate text-model fusion weights (0.63 for TF-IDF and 0.38 for "
        "DistilBERT) were selected by binary F1 for label 1 on the 150-example "
        "validation subset. The fixed weights were evaluated on 150 aligned "
        "official test examples. This is exploratory and not full-test "
        "multimodal performance."
    )
    st.markdown("### Split discipline")
    st.write(
        "The saved DistilBERT and ResNet checkpoints were selected using "
        "validation data. The fusion rule selects its weights on validation. "
        "The final fusion test subset is loaded only after that selection. "
        "Official train/validation/test splits remain separate."
    )


def render_limitations() -> None:
    hero(
        "Limitations & responsible use",
        "A trained dataset classifier is not an independent fact-checking authority.",
    )
    st.markdown(
        """
        - **No ground-truth adjudication:** predictions reflect learned Fakeddit label patterns, not live source verification.
        - **Scores are not calibrated truth probabilities:** read them as model class scores.
        - **Fusion is exploratory:** weights were selected on a 150-example validation subset and tested on 150 aligned examples.
        - **Different evaluation scopes:** text results use 59,163 test rows; ResNet and fusion use 150 held-out image examples.
        - **Context modeling is incomplete:** raw domain/subreddit fields exist but were not used as model inputs.
        - **CLIP is unavailable:** no local CLIP checkpoint or measured image-text consistency component is included.
        - **Dataset bias and duplicates:** class imbalance and possible semantic duplicates under different IDs can affect generalization.
        - **Local computation:** all supplied model inference runs on CPU; larger transformer inference may take time.
        """
    )
    st.error(
        "Do not use this demonstration to make consequential decisions about "
        "people, organizations, elections or public safety. Verify claims with "
        "independent, reliable sources."
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## 🧭 Fakeddit")
        st.caption("Context-aware multimodal misinformation project")
        page = st.radio(
            "Project navigation",
            (
                "Home",
                "Analyze a claim",
                "Saved results",
                "Methodology",
                "Limitations",
            ),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Labels: 0 = Fake · 1 = True")
        st.caption("Local inference · no remote image uploads")
    if page == "Home":
        render_home()
    elif page == "Analyze a claim":
        render_analysis()
    elif page == "Saved results":
        render_results()
    elif page == "Methodology":
        render_methodology()
    else:
        render_limitations()
    st.divider()
    st.caption(
        "B.Tech CSE major-project demonstrator · Fakeddit · "
        "Research output, not a fact-check verdict."
    )


if __name__ == "__main__":
    main()
