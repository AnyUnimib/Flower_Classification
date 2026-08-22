# Customer Churn Prediction — Spark ML Pipeline

**Any Das**

An end-to-end, production-style customer churn prediction pipeline built with **PySpark / Spark MLlib**, benchmarking 5 classification algorithms across two datasets with different class distributions (balanced vs. heavily imbalanced).

The goal wasn't just to hit a good accuracy number — it was to build a pipeline that generalizes across data distributions and stays interpretable enough to turn into a retention strategy.

## Why this project

Churn prediction is a classic case where the "best" model on paper isn't always the most useful one in practice — imbalanced classes, interpretability requirements, and the cost of false negatives (losing a customer you didn't flag) all matter more than raw accuracy. This project was built to explore that trade-off directly, by running the same pipeline on a balanced and an imbalanced dataset and comparing how 5 different algorithms behave under each.

## Datasets

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Records | 40,000 | 50,000 |
| Class balance | 51.8% churned / 48.2% retained (balanced) | 87.3% churned / 12.7% retained (imbalanced) |
| Features | 13 (e.g. `monthly_spend`, `last_campaign_days`, `tenure_months`, `email_click_rate`) | same |
| Split | 80% train / 20% test | 80% train / 20% test |

Running the same pipeline on both let me isolate how much of the performance was coming from the model vs. the underlying data distribution.

## Pipeline Architecture

Everything is built as composable Spark `Pipeline` stages, so preprocessing is fit once and reused identically across all 5 models (no leakage, no duplicated logic):

```
Raw CSV
  └─ Load + assign row_id (preserves original order for prediction export)
      └─ Train/test split (80/20, seeded)
          └─ Preprocessing Pipeline
               ├─ Numerical:   Imputer (median) → VectorAssembler → StandardScaler
               └─ Categorical: StringIndexer → OneHotEncoder
                    └─ VectorAssembler (final feature vector)
                        └─ Classifier (one of 5)
                            └─ Evaluation + Feature Importance + Prediction Export
```

**Design choices worth calling out:**
- **Median imputation** over mean — more robust to skewed spend/engagement distributions.
- **`handleInvalid="keep"`** throughout — unseen categories/nulls at inference time don't crash the pipeline, they route to a reserved bucket.
- **Preprocessor as a Spark `Pipeline` stage, not a standalone step** — it gets re-fit inside each model's pipeline, guaranteeing train/test consistency without manual state management.
- **`row_id` tracking** — since Spark doesn't guarantee row order, a monotonically increasing ID is attached at load time so predictions can be joined back to the original dataset in order.

### Models compared
Logistic Regression · Random Forest · Gradient Boosted Trees · Decision Tree · Linear SVM

### Evaluation
Accuracy, Precision, Recall, F1, ROC-AUC (via Spark's built-in evaluators), plus confusion matrices, ROC curves, and Precision-Recall curves plotted per dataset for direct model comparison.

## Results

### Dataset 1 — Balanced (best model: Logistic Regression)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Retained (0) | 0.81 | 0.80 | 0.80 |
| Churned (1) | 0.81 | 0.82 | 0.82 |
| **Overall accuracy** | | | **0.811** |

SVM tracked Logistic Regression closely across all metrics. Gradient Boosting and Random Forest trailed slightly behind, and Decision Tree was consistently the weakest model on both the ROC and Precision-Recall curves.

### Dataset 2 — Imbalanced (best model: Logistic Regression)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Retained (0) | 0.80 | 0.70 | 0.75 |
| Churned (1) | 0.96 | 0.97 | 0.97 |
| **Overall accuracy** | | | **0.94** |

All 5 models handled the imbalance reasonably well and detected churn effectively; Random Forest had the fewest false negatives, while Logistic Regression and SVM kept the best balance between false positives and false negatives overall.

### What stood out

- **Simple beat complex.** Logistic Regression and Linear SVM consistently matched or outperformed the tree-based ensembles on both datasets — the decision boundary here is close to linear once features are properly scaled and encoded.
- **`monthly_spend` and `last_campaign_days` dominate.** Across every model and both datasets, these two features carried the most predictive weight — spending behavior and recency of engagement matter far more than static demographics.
- **Imbalance didn't break the pipeline, but it did shift the trade-off.** On the 87/13 imbalanced dataset, every model got very good at catching churners (recall ~0.97) but noticeably worse at correctly identifying customers who'd stay (recall ~0.70) — the kind of asymmetry that matters when deciding whether false positives (wrongly flagging a loyal customer) are cheap or expensive for the business.
- **Tree-based models were "narrower."** Random Forest and Decision Tree leaned almost entirely on 2 features, while Logistic Regression, SVM, and GBT spread importance across `tenure_months`, `loyalty_score`, `email_click_rate`, and `website_visits` — a more complete picture of customer behavior.

## Repository Structure

```
├── HW3.py                           # Full pipeline: load → preprocess → train → evaluate → export
├── dataset1_HW1.csv                 # Balanced dataset
├── dataset2_HW1.csv                 # Imbalanced dataset
├── predictions_dataset1/            # Train/test/full prediction CSVs
├── predictions_dataset2/
├── visualizations_dataset1/         # ROC/PR curves, confusion matrices, feature importance plots
└── visualizations_dataset2/
```

## Running it

```bash
pip install pyspark numpy matplotlib seaborn
python HW3.py
```

Runs the full pipeline sequentially for both datasets — trains all 5 models, prints evaluation metrics to console, saves plots and predictions to disk.

## What I'd improve next

- Add SMOTE or class weighting to see if recall on the minority (retained) class in Dataset 2 improves without sacrificing overall performance.
- Hyperparameter tuning via `CrossValidator` / `ParamGridBuilder` (current run uses default/lightly-tuned params to keep the comparison apples-to-apples).
- Package the pipeline as a reusable module with a config file instead of hardcoded dataset paths.
- Add SHAP-style explanations for a more rigorous feature attribution than native importances/coefficients.

## Tech Stack
Python · PySpark (Spark MLlib) · Matplotlib · Seaborn
