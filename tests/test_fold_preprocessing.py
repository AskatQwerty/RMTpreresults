"""Тесты политики пропусков исключительно на искусственных данных."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.fold_preprocessing import make_numeric_pipeline, numeric_pipeline_smoke_check


class FoldSafeNumericPipelineTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "empty": [np.nan, np.nan, np.nan],
            "partial": [1.0, np.nan, 3.0],
            "complete": [10.0, 20.0, 30.0],
        })
        self.validation = pd.DataFrame({
            "empty": [7.0, np.nan],
            "partial": [np.nan, 999.0],
            "complete": [np.nan, 40.0],
        })

    def test_factory_does_not_fit(self):
        pipeline = make_numeric_pipeline()
        with self.assertRaises(NotFittedError):
            pipeline.transform(self.train)

    def test_all_empty_train_column_kept_and_validation_observed_preserved(self):
        pipeline = make_numeric_pipeline()
        fitted = pipeline.fit_transform(self.train)
        transformed = pipeline.transform(self.validation)
        self.assertEqual(fitted.shape, (3, 6))
        self.assertEqual(transformed.shape, (2, 6))
        np.testing.assert_array_equal(fitted[:, 0], [0, 0, 0])
        np.testing.assert_array_equal(transformed[:, 0], [7, 0])
        np.testing.assert_array_equal(transformed[:, 3], [0, 1])

    def test_median_uses_only_training_values(self):
        pipeline = make_numeric_pipeline().fit(self.train)
        transformed = pipeline.transform(self.validation)
        self.assertEqual(transformed[0, 1], 2.0)
        self.assertEqual(transformed[1, 1], 999.0)
        imputer = pipeline.named_steps["numeric"].transformer_list[0][1].named_steps["median"]
        np.testing.assert_array_equal(imputer.statistics_, [0.0, 2.0, 20.0])
        # Extreme held-out values do not modify learned statistics on transform.
        pipeline.transform(self.validation * 1000000)
        np.testing.assert_array_equal(imputer.statistics_, [0.0, 2.0, 20.0])

    def test_new_validation_missing_gets_indicator(self):
        pipeline = make_numeric_pipeline().fit(self.train)
        transformed = pipeline.transform(self.validation)
        self.assertEqual(transformed[0, 2], 20.0)
        self.assertEqual(transformed[0, 5], 1.0)
        self.assertEqual(transformed[1, 5], 0.0)

    def test_folds_have_same_width_names_even_when_missingness_changes(self):
        first = make_numeric_pipeline().fit(self.train)
        second_train = self.train.copy()
        second_train["empty"] = [2.0, 4.0, 6.0]
        second_train["partial"] = [np.nan, np.nan, np.nan]
        second = clone(first).fit(second_train)
        np.testing.assert_array_equal(first.get_feature_names_out(), second.get_feature_names_out())
        self.assertEqual(len(set(first.get_feature_names_out())), 6)
        first_out = first.transform(self.validation)
        second_out = second.transform(self.validation)
        self.assertEqual(first_out.shape, second_out.shape)
        self.assertEqual(first_out[1, 0], 0)
        self.assertEqual(second_out[1, 0], 4)
        self.assertEqual(first_out[0, 1], 2)
        self.assertEqual(second_out[0, 1], 0)

    def test_all_numeric_columns_empty(self):
        frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        pipeline = make_numeric_pipeline().fit(frame)
        transformed = pipeline.transform(frame)
        np.testing.assert_array_equal(transformed, [[0, 0, 1, 1], [0, 0, 1, 1]])

    def test_scaler_fitted_only_values_and_empty_column_safe(self):
        pipeline = make_numeric_pipeline(scale=True).fit(self.train)
        transformed = pipeline.transform(self.validation)
        scaler = pipeline.named_steps["numeric"].transformer_list[0][1].named_steps["scale"]
        np.testing.assert_array_equal(scaler.mean_, [0, 2, 20])
        self.assertEqual(scaler.scale_[0], 1)
        np.testing.assert_array_equal(transformed[:, 0], [7, 0])
        np.testing.assert_array_equal(transformed[:, 3:], [[0, 1, 1], [1, 0, 0]])
        self.assertTrue(np.isfinite(transformed).all())

    def test_clone_is_unfitted_and_original_input_unchanged(self):
        copy = self.train.copy(deep=True)
        fitted = make_numeric_pipeline().fit(self.train)
        cloned = clone(fitted)
        with self.assertRaises(NotFittedError):
            cloned.transform(self.validation)
        pd.testing.assert_frame_equal(self.train, copy)

    def test_smoke_report_has_no_real_data_fit(self):
        report = numeric_pipeline_smoke_check()
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["scope"], "synthetic_only_no_fit_on_home_credit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
