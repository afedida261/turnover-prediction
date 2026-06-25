import unittest

import pandas as pd

from src.imputations import SimilarEmployeePaymentImputer
from src.preprocess import add_time_safe_history_features, clean_problematic_records


def _base_rows():
    return pd.DataFrame(
        {
            "source": ["file1", "file1", "file1"],
            "source_employee_id": ["file1:1"] * 3,
            "fictive_employee": [1] * 3,
            "calc_month": pd.to_datetime(["2021-12-01", "2022-12-01", "2023-12-01"]),
            "leave_ind": [0, 0, 1],
            "age": [76, 77, 78],
            "vetek_months": [12, 24, 36],
            "avg_Payment": [100.0, 200.0, 300.0],
            "Median_Payment": [100.0, 200.0, 300.0],
            "stdevp_Payment": [0.0, 10.0, 20.0],
            "salary_change": [0.0, 100.0, 100.0],
            "avg_illness": [1.0, 2.0, 3.0],
            "Median_illness": [0.0, 1.0, 2.0],
            "stdevp_illness": [1.0, 1.0, 1.0],
            "avg_omes": [1.0, 1.0, 1.0],
            "Median_omes": [1.0, 1.0, 1.0],
            "stdevp_omes": [0.1, 0.1, 0.1],
            "count_managers": [0.0, 1.0, 1.0],
        }
    )


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_retains_age_above_75_and_removes_confirmed_invalid_rows(self):
        valid_old_employee = _base_rows().iloc[[0]].copy()
        invalid = valid_old_employee.copy()
        invalid["age"] = 17
        frame = pd.concat([valid_old_employee, invalid], ignore_index=True)

        cleaned, audit = clean_problematic_records(frame)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["age"], 76)
        self.assertEqual(audit.retained_age_above_75_rows, 1)
        self.assertEqual(audit.reason_counts["age_below_18"], 1)

    def test_history_features_do_not_use_future_values(self):
        full = add_time_safe_history_features(_base_rows())
        prefix = add_time_safe_history_features(_base_rows().iloc[:2].copy())

        columns = [
            "history_prior_records",
            "avg_Payment_hist_prev",
            "avg_Payment_hist_prior_mean",
            "avg_Payment_hist_delta_prev",
        ]
        pd.testing.assert_frame_equal(
            full.iloc[:2][columns].reset_index(drop=True),
            prefix[columns].reset_index(drop=True),
        )

    def test_payment_imputer_refuses_external_test_data_during_fit(self):
        frame = _base_rows().assign(source="file3", contract_type="Monthly")
        with self.assertRaisesRegex(ValueError, "external test source"):
            SimilarEmployeePaymentImputer(min_observed_rows=1).fit(frame)


if __name__ == "__main__":
    unittest.main()

