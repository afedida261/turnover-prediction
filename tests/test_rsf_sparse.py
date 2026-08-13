import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.analysis.random_survival_forest import (
    fit_final_rsf_with_stability,
    make_rsf_preprocessor,
)


class RSFOOBMatrixCompatibilityTests(unittest.TestCase):
    def test_preprocessor_and_oob_fit_avoid_sparse_oob_bug(self):
        rows = 40
        X = pd.DataFrame(
            {
                "source": ["file1"] * rows,
                "year_date": [2022] * rows,
                "age": np.arange(rows) + 20,
                "contract_type": np.where(np.arange(rows) % 2, "Monthly", "Hourly"),
            }
        )
        y = np.array(
            list(zip(np.arange(rows) % 3 == 0, np.linspace(2.0, 30.0, rows))),
            dtype=[("event", "?"), ("time", "<f8")],
        )
        transformed = make_rsf_preprocessor(X).fit_transform(X, y)
        self.assertFalse(sparse.issparse(transformed))
        self.assertEqual(transformed.dtype, np.float32)

        model, _ = fit_final_rsf_with_stability(
            sparse.csc_matrix(transformed),
            y,
            params={
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "max_features": "sqrt",
                "max_depth": None,
            },
            tree_counts=[10],
            n_jobs=1,
        )
        self.assertEqual(model.n_estimators, 10)


if __name__ == "__main__":
    unittest.main()
