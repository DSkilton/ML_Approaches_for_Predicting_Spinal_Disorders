import unittest
import numpy as np
import pandas as pd
from evaluation import TARGET, LABEL, regression_inputs, classification_inputs, evaluate_classification, evaluate_regression

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.data = pd.DataFrame({
            "pelvic_incidence": rng.normal(50, 10, 90),
            "pelvic_tilt": rng.normal(20, 5, 90),
            "lumbar_lordosis_angle": rng.normal(40, 10, 90),
            "sacral_slope": rng.normal(30, 6, 90),
            "pelvic_radius": rng.normal(100, 12, 90),
            TARGET: rng.normal(8, 3, 90),
            LABEL: ["normal", "hernia", "spondylolisthesis"] * 30,
        })

    def test_target_leakage_guard(self):
        X, y = regression_inputs(self.data)
        self.assertNotIn(TARGET, X)
        self.assertNotIn(LABEL, X)
        self.assertEqual(len(X), len(y))

    def test_hierarchical_labels_become_noncontradictory_binary_target(self):
        _, y = classification_inputs(self.data)
        expected = ["normal" if x == "normal" else "pathology" for x in self.data[LABEL]]
        self.assertEqual(y.tolist(), expected)

    def test_repeated_measurement_vectors_are_deduplicated(self):
        copied = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        X, _ = classification_inputs(copied)
        self.assertEqual(len(X), 90)
        X_reg, _ = regression_inputs(copied)
        self.assertEqual(len(X_reg), 90)

    def test_contradictory_binary_labels_are_rejected(self):
        contradictory = self.data.iloc[[1]].copy()
        contradictory[LABEL] = "normal"
        copied = pd.concat([self.data, contradictory], ignore_index=True)
        with self.assertRaises(ValueError):
            classification_inputs(copied)

    def test_holdout_models(self):
        cls = evaluate_classification(self.data, 3)
        reg = evaluate_regression(self.data, 3)
        self.assertIn("test_macro_f1", cls)
        self.assertIn("test_r2", reg)
        self.assertNotIn(TARGET, reg["features"])
        self.assertEqual(cls["test_rows"], 18)

if __name__ == "__main__":
    unittest.main()
