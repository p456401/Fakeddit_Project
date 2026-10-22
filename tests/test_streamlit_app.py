from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


def render_page(page: str = "Home") -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    if page != "Home":
        app.radio[0].set_value(page).run()
    return app


class StreamlitApplicationTests(unittest.TestCase):
    def test_home_page_renders_without_exceptions(self) -> None:
        app = render_page()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "A transparent, model-by-model workflow" in item.value
                for item in app.markdown
            )
        )

    def test_saved_results_page_renders_table_and_confusion_matrix_selector(self) -> None:
        app = render_page("Saved results")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertIn("TF-IDF + Logistic Regression", app.selectbox[0].options)

    def test_blank_analysis_is_rejected_before_model_loading(self) -> None:
        app = render_page("Analyze a claim")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Enter text or upload a valid local image" in item.value
                for item in app.warning
            )
        )


if __name__ == "__main__":
    unittest.main()
