"""Protect the recorded notebook correction against accidental reintroduction."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        cls.cells = ["".join(c.get("source", [])) for c in notebook["cells"]]

    def test_regression_does_not_include_its_target(self):
        source = next(s for s in self.cells if "X_regression =" in s)
        self.assertIn("assert DEGREE_SPONDYLOLISTHESIS not in X_regression.columns", source)
        self.assertNotIn("X_regression = cleaned_data.drop(", source)

    def test_heuristic_labels_do_not_override_original_data(self):
        source = next(s for s in self.cells if "def classify_row(" in s)
        self.assertNotIn("unclean_data = updated_df", source)
        self.assertIn("original unclean_data unchanged", source)

    def test_ablation_imputer_constructor_uses_keyword(self):
        source = next(s for s in self.cells if "ablations = {" in s)
        self.assertNotIn("KNNImputer(5)", source)
        self.assertIn("KNNImputer(n_neighbors=5)", source)

if __name__ == "__main__":
    unittest.main()
