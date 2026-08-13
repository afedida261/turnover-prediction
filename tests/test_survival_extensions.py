import unittest

import numpy as np

from src.analysis.random_survival_forest import (
    DEFAULT_RSF_PARAMETERS,
    build_rsf_tuning_grid,
)
from src.analysis.survival_extensions import (
    DETERMINISTIC_REDUNDANT_COLUMNS,
    PooledLogisticSurvival,
    calibration_table,
)


def _target(events, times):
    return np.array(
        list(zip(events, times)),
        dtype=[("event", "?"), ("time", "<f8")],
    )


class SurvivalExtensionTests(unittest.TestCase):
    def test_rsf_tuning_grid_is_reproducible_and_derives_split_size(self):
        first = build_rsf_tuning_grid(max_candidates=24, seed=42)
        second = build_rsf_tuning_grid(max_candidates=24, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(first[0], DEFAULT_RSF_PARAMETERS)
        self.assertTrue(
            all(
                candidate["min_samples_split"] == 2 * candidate["min_samples_leaf"]
                for candidate in first
            )
        )

    def test_zero_candidate_limit_requests_full_grid(self):
        self.assertEqual(len(build_rsf_tuning_grid(max_candidates=0, seed=42)), 144)

    def test_known_deterministic_redundancies_are_declared(self):
        self.assertEqual(
            DETERMINISTIC_REDUNDANT_COLUMNS,
            {"tenure_years", "career_start_age", "is_new_employee"},
        )

    def test_calibration_table_uses_km_observed_risk(self):
        y = _target(
            [True, False, True, False, True, False, True, False],
            [3.0, 12.0, 5.0, 12.0, 8.0, 12.0, 10.0, 12.0],
        )
        survival = np.column_stack(
            [
                np.linspace(0.95, 0.55, len(y)),
                np.linspace(0.85, 0.35, len(y)),
            ]
        )
        table = calibration_table(
            y,
            survival,
            [6.0, 12.0],
            n_bins=2,
            bootstrap_repeats=5,
        )
        self.assertEqual(sorted(table["Horizon_Months"].unique().tolist()), [6.0, 12.0])
        self.assertIn("Observed_Risk_KM", table.columns)
        self.assertIn("Observed_Risk_CI_Lower", table.columns)
        self.assertIn("Bin_Calibration_Difference", table.columns)

    def test_tied_predictions_fall_back_to_one_calibration_bin(self):
        y = _target(
            [True, False, True, False],
            [3.0, 12.0, 8.0, 12.0],
        )
        survival = np.full((len(y), 2), 0.75)
        table = calibration_table(
            y,
            survival,
            [6.0, 12.0],
            n_bins=5,
            bootstrap_repeats=2,
        )
        self.assertEqual(len(table), 2)
        self.assertEqual(table["Risk_Bin"].tolist(), [1, 1])
        self.assertEqual(table["Employees"].tolist(), [4, 4])

    def test_pooled_logistic_returns_monotone_survival_curves(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.2, 1.0],
                [0.4, 0.0],
                [0.6, 1.0],
                [0.8, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = _target([True, False, True, False, True, False], [3.0, 12.0, 7.0, 12.0, 10.0, 12.0])
        model = PooledLogisticSurvival(interval_months=3.0, max_iter=200).fit(X, y)
        curves = model.predict_survival_function(X)
        for curve in curves:
            values = curve(np.array([0.0, 3.0, 6.0, 9.0, 12.0]))
            self.assertAlmostEqual(values[0], 1.0)
            self.assertTrue(np.all(np.diff(values) <= 0.0))
        self.assertEqual(model.predict(X).shape, (len(X),))


if __name__ == "__main__":
    unittest.main()
