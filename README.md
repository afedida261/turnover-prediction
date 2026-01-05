# Turnover Prediction

ML project aiming to predict employee turnover in organizations using synthetic data based on Hilan requirements.

## Project Overview

This project implements a modular machine learning pipeline to predict employee churn. It includes:
- Synthetic data generation matching specific Hebrew field requirements.
- Data preprocessing and feature engineering.
- Multiple model implementations:
  - Traditional Classifiers (Random Forest, XGBoost, Logistic Regression)
  - Neural Networks (PyTorch)
  - Survival Analysis (CoxPH)
- Evaluation metrics focused on business impact (Recall@TopK, Lift).

## Project Structure

```
turnover-prediction/
├── data/               # Data storage
├── src/                # Source code
│   ├── models/         # Model implementations
│   ├── config.py       # Configuration and mappings
│   ├── data_loader.py  # Data loading and processing
│   ├── evaluator.py    # Evaluation metrics
│   └── generate_data.py # Synthetic data generator
├── main.py             # Main entry point
└── PROJECT_PLAN.md     # Detailed project plan
```

## Usage

1. **Generate Data:**
   ```bash
   python src/generate_data.py
   ```

2. **Run Pipeline:**
   ```bash
   python main.py
   ```
