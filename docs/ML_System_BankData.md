# Machine Learning System on Bank Statement Data

## ML Systems You Can Build

### 1. Personal Finance Anomaly Detection
Detect unusual transactions (potential fraud or overspending).
- **Model:** Isolation Forest, Autoencoder, or One-Class SVM
- **Features:** transaction amount, day of week, merchant category, frequency
- **Label:** anomaly score (unsupervised) or flagged/normal (if you manually label some)

### 2. Spending Category Classifier
Auto-classify transactions into categories (food, transport, rent, etc.).
- **Model:** Random Forest, Gradient Boosting (XGBoost/LightGBM)
- **Features:** transaction description (NLP), amount, date
- **Label:** category (you define and label manually)

### 3. Monthly Cash Flow Forecasting
Predict next month's balance or spending.
- **Model:** Gradient Boosting Regressor, LSTM (if going deep learning)
- **Features:** rolling averages, month, income/expense trends
- **Target:** end-of-month balance or total spend

### 4. Bank Creditworthiness Scoring *(most impressive for recruiters)*
Simulate what a bank does — score your own financial health.
- **Model:** Logistic Regression + Random Forest + XGBoost (ensemble)
- **Features:** DSCR, savings rate, expense volatility, overdraft frequency
- **Target:** credit score tier (Low / Medium / High risk)

---

## Recommended Project: Spending Classifier + Creditworthiness Score

This covers the most ML fundamentals and tells a clear story.

---

## Key Steps (Mapped to Your Data)

```
1. DATA COLLECTION
   └── Parse PDFs → extract transactions (date, description, amount, balance)

2. DATA PREPARATION
   ├── Clean: handle missing values, normalize amounts
   ├── Engineer features:
   │     - transaction type (debit/credit)
   │     - rolling 30/60/90 day spend
   │     - savings rate per month
   │     - overdraft flag
   │     - day of week / month
   └── Encode categoricals (label encoding or one-hot)

3. LABELING
   └── Manually tag ~200 transactions with categories
       (or use keyword rules to auto-label as baseline)

4. SPLIT DATA
   ├── Train:      70%
   ├── Validation: 15%  ← tune hyperparameters here
   └── Test:       15%  ← final evaluation only

5. TRAIN MODELS
   ├── Baseline: Logistic Regression
   ├── Intermediate: Random Forest
   └── Advanced: XGBoost / LightGBM

6. EVALUATE
   ├── Classification: Accuracy, F1-score, Confusion Matrix, ROC-AUC
   └── Regression: MAE, RMSE, R²

7. VISUALIZE & REPORT
   └── Feature importance, monthly trends, model comparison charts
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| PDF Parsing | `pdfplumber` or `camelot` |
| Data Processing | `pandas`, `numpy` |
| ML Models | `scikit-learn`, `xgboost` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Notebook | `Jupyter` |
| Optional Dashboard | `Streamlit` |

---

## What This Demonstrates to Recruiters

- **Data engineering** — parsing real-world messy PDFs
- **Feature engineering** — creating meaningful signals from raw transactions
- **ML pipeline** — train/val/test split, cross-validation
- **Model selection** — comparing multiple algorithms
- **Evaluation** — using the right metrics, not just accuracy
- **Domain knowledge** — understanding financial data like a data scientist

---

## Purpose

Show recruiters you understand **core ML fundamentals** like data preprocessing, training, and evaluation.
