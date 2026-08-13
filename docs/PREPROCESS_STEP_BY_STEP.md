# `src/preprocess.py` Step-by-Step

1. Defines the source files, target column, and the columns that should always be excluded from modeling.
2. Normalizes raw input by renaming known column aliases and replacing Hebrew category values with English labels.
3. Validates that the loaded data contains the required metadata columns and that target values are only `0` or `1`.
4. Removes records that were confirmed invalid by the EDA rules, while keeping an audit of what was removed.
5. Builds time-safe history features for each employee using only the current row and earlier rows.
6. Adds extra EDA-driven engineered features such as salary gaps, coefficients of variation, tenure features, and manager-change rates.
7. Loads the Excel sources, tags each row with its source, and combines all sources into one frame.
8. Selects model columns by dropping outcome fields, auxiliary fields, protected/excluded fields, and date columns.
9. Splits the cleaned data into train and test sets by source, then returns frames, labels, employee groups, audit data, and dropped-column metadata.
10. Builds a fold-safe preprocessing pipeline that imputes payment features, imputes numeric and categorical values, scales numeric data, and one-hot encodes categorical data.
11. Fits the preprocessing pipeline on the training set and saves the transformer, cleaned train/test frames, and JSON metadata.
12. Provides a CLI entry point that prints the cleaning summary and row counts, or saves artifacts unless `--dry-run` is used.
