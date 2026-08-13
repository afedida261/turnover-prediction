import unittest

import numpy as np
import pandas as pd

from src.analysis.random_survival_forest import (
    administratively_truncate_survival,
    build_employee_survival_cohort,
    choose_evaluation_times,
    fit_final_rsf_with_stability,
)


def _survival_rows():
    return pd.DataFrame(
        {
            "source": ["file1"] * 6,
            "source_employee_id": ["file1:1", "file1:1", "file1:2", "file1:3", "file1:3", "file1:3"],
            "calc_month": pd.to_datetime(
                ["2022-01-01", "2023-01-01", "2022-06-15", "2021-01-01", "2022-01-01", "2023-01-01"]
            ),
            "aziva_date": pd.to_datetime(
                [None, "2023-04-01", None, None, "2022-03-01", "2023-04-01"]
            ),
            "leave_ind": [0, 1, 0, 0, 1, 1],
            "age": [30, 31, 40, 50, 51, 52],
            "avg_Payment": [100, 110, 200, 300, 310, 320],
            "year_date": [2022, 2023, 2022, 2021, 2022, 2023],
        }
    )


class SurvivalCohortTests(unittest.TestCase):
    def test_one_baseline_row_per_employee_with_exact_event_and_twelve_month_censoring(self):
        cohort = build_employee_survival_cohort(
            _survival_rows(),
            feature_columns=["source", "year_date", "age", "avg_Payment"],
        )

        self.assertEqual(len(cohort.X), 3)
        self.assertFalse(cohort.metadata["source_employee_id"].duplicated().any())
        self.assertEqual(cohort.X["age"].tolist(), [30, 40, 50])

        employee_1 = cohort.metadata.loc[cohort.metadata["source_employee_id"].eq("file1:1")].iloc[0]
        self.assertTrue(employee_1["event"])
        self.assertAlmostEqual(employee_1["observed_time_months"], 15.0, delta=0.10)

        employee_2 = cohort.metadata.loc[cohort.metadata["source_employee_id"].eq("file1:2")].iloc[0]
        self.assertFalse(employee_2["event"])
        self.assertAlmostEqual(employee_2["observed_time_months"], 12.0, delta=0.10)

        employee_3 = cohort.metadata.loc[cohort.metadata["source_employee_id"].eq("file1:3")].iloc[0]
        self.assertEqual(employee_3["event_date"], pd.Timestamp("2022-03-01"))
        self.assertEqual(cohort.audit.multiple_event_employees, 1)
        self.assertEqual(cohort.audit.post_event_rows_ignored, 1)
        self.assertEqual(cohort.y.dtype.names, ("event", "time"))

    def test_outcome_columns_are_rejected_as_covariates(self):
        with self.assertRaisesRegex(ValueError, "cannot be covariates"):
            build_employee_survival_cohort(
                _survival_rows(),
                feature_columns=["age", "aziva_date"],
            )

    def test_missing_event_date_is_rejected_in_strict_mode(self):
        frame = _survival_rows().iloc[[0]].copy()
        frame["leave_ind"] = 1
        with self.assertRaisesRegex(ValueError, "missing event dates=1"):
            build_employee_survival_cohort(frame, feature_columns=["age"])


class SurvivalMetricTests(unittest.TestCase):
    @staticmethod
    def _target(events, times):
        return np.array(
            list(zip(events, times)),
            dtype=[("event", "?"), ("time", "<f8")],
        )

    def test_administrative_truncation_censors_later_events(self):
        y = self._target([True, False, True], [5.0, 20.0, 30.0])
        truncated = administratively_truncate_survival(y, tau=15.0)
        self.assertEqual(truncated["event"].tolist(), [True, False, False])
        self.assertEqual(truncated["time"].tolist(), [5.0, 15.0, 15.0])

    def test_evaluation_times_stay_inside_common_support(self):
        y_train = self._target([True, False, True, False], [3.0, 12.0, 24.0, 40.0])
        y_test = self._target([True, False, True, False], [4.0, 12.0, 20.0, 40.0])
        grid, reported, tau = choose_evaluation_times(y_train, y_test, grid_points=8)
        self.assertGreaterEqual(grid.min(), 1.0)
        self.assertLess(grid.max(), 40.0)
        self.assertLess(tau, 40.0)
        self.assertTrue(set(reported).issubset({6.0, 12.0, 24.0, 36.0}))

    def test_small_forest_smoke_fit(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 4)).astype(np.float32)
        y = self._target(
            [index % 3 == 0 for index in range(40)],
            np.linspace(2.0, 30.0, num=40),
        )
        model, stability = fit_final_rsf_with_stability(
            X,
            y,
            params={
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "max_features": "sqrt",
                "max_depth": None,
            },
            tree_counts=[10, 20],
            n_jobs=1,
        )
        self.assertEqual(model.n_estimators, 20)
        self.assertEqual(stability["Trees"].tolist(), [10, 20])
        self.assertTrue(np.isfinite(stability["OOB_Harrell_C"]).all())


if __name__ == "__main__":
    unittest.main()
