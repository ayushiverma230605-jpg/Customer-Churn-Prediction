# Customer-Churn-Prediction
# ML Classification Project — Task 1 Report
## Customer Churn Prediction
**Alfido Tech Internship** | June 2026

---

## 1. Problem Statement

Customer churn is a critical business metric for telecom companies. Identifying customers likely to churn enables targeted retention strategies, reducing revenue loss. This project builds a binary classification system (Churn: Yes / No) using historical customer data.

---

## 2. Dataset

| Attribute | Value |
|-----------|-------|
| Source | Synthetic Telco Churn (reproducible, `random_state=42`) |
| Samples | 1,500 |
| Features | 15 (tenure, charges, services, contract type, etc.) |
| Target | Churn (binary) |
| Class ratio | 72% No Churn / 28% Churn (realistic imbalance) |
| Missing values | ~2.5% in `total_charges`, `tech_support`, `online_security` |

---

## 3. Methodology

### 3.1 Data Preprocessing
- **Missing values:** Median imputation via `sklearn.impute.SimpleImputer` inside each pipeline — prevents data leakage during cross-validation.
- **Feature scaling:** `StandardScaler` applied for Logistic Regression and Gradient Boosting.
- **Class imbalance:** Handled via `class_weight='balanced'` in Logistic Regression and Random Forest.

### 3.2 Train / Test Split
Stratified 80/20 split ensures identical churn ratios in both sets.

| Set | Samples | Churn Rate |
|-----|---------|------------|
| Train | 1,200 | 28.7% |
| Test | 300 | 28.7% |

### 3.3 Cross-Validation
5-Fold Stratified K-Fold on the training set; optimized for **ROC-AUC** (preferred over accuracy for imbalanced classification).

---

## 4. Models Compared

Three algorithms were evaluated end-to-end via sklearn Pipelines:

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline; fast, interpretable, `class_weight='balanced'` |
| Random Forest | Ensemble of 200 decision trees; handles non-linearity, built-in feature importance |
| Gradient Boosting | Boosted ensemble; 150 trees, `lr=0.08`, subsampling; typically strongest |

---

## 5. Results

### 5.1 Cross-Validation (5-Fold ROC-AUC on Training Set)

| Model | Mean AUC | Std |
|-------|----------|-----|
| Logistic Regression | 0.8994 | ±0.0220 |
| Random Forest | 0.9264 | ±0.0178 |
| **Gradient Boosting** | **0.9316** | **±0.0197** |

### 5.2 Test Set Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 87.3% | 0.745 | 0.849 | 0.794 | 0.918 |
| Random Forest | 88.7% | 0.861 | 0.721 | 0.785 | 0.927 |
| **Gradient Boosting** | **90.7%** | **0.903** | **0.756** | **0.823** | **0.930** |

### 5.3 Key Observations

- **Logistic Regression** has the highest recall (0.849) — fewest missed churners — but lower precision.
- **Random Forest** offers balanced performance with the most interpretable feature importances.
- **Gradient Boosting** achieves the best accuracy (90.7%), F1 (0.823), and ROC-AUC (0.930), making it the clear winner overall.

---

## 6. Model Selection: Gradient Boosting ✅

**Recommendation:** Deploy Gradient Boosting for production churn prediction.

**Rationale:**
- Highest ROC-AUC (0.930) — best discrimination between churners and non-churners.
- Highest F1 (0.823) — best balance between precision and recall.
- Highest accuracy (90.7%) on held-out test data.
- Robust to feature scaling differences and mild non-linearities.

**Trade-off note:** If recall (catching every churner) is paramount (e.g., retention budget is unlimited), Logistic Regression's higher recall (0.849 vs 0.756) may be preferred at the cost of more false positives.

---

## 7. Feature Importance (Top 5 — Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | total_charges | Highest |
| 2 | monthly_charges | High |
| 3 | tenure_months | High |
| 4 | num_services | Medium |
| 5 | contract_type_enc | Medium |

**Business Insight:** Customers with high monthly charges and low tenure are the highest churn risk. Targeted offers (discounts, loyalty rewards) for this segment would be most effective.

---

## 8. Responsible AI Considerations

| Consideration | Approach Taken |
|---------------|---------------|
| Class imbalance | `class_weight='balanced'`, stratified splits |
| Data leakage | Imputation inside Pipeline — no test data seen during training |
| Reproducibility | `random_state=42` throughout; fixed package versions |
| Fairness | Feature set excludes protected attributes (age, gender, ethnicity) |
| Interpretability | Feature importance from RF; LR coefficients interpretable |

---

## 9. Plots Generated

| # | Plot | Purpose |
|---|------|---------|
| 1 | Feature distributions by churn | EDA — understand signal |
| 2 | Class balance (bar + pie) | Quantify imbalance |
| 3 | Correlation heatmap | Feature collinearity |
| 4 | CV ROC-AUC boxplot | Model stability comparison |
| 5 | ROC curves (all models) | Discrimination ability |
| 6 | Metrics bar chart | Side-by-side comparison |
| 7 | Confusion matrices | Error breakdown |
| 8 | Feature importances (RF) | Top predictors |
| 9 | Probability distribution | Calibration insight |

---

## 10. Conclusion

Gradient Boosting is the recommended model for customer churn prediction, delivering 90.7% accuracy and a ROC-AUC of 0.930. The model is well-calibrated, avoids data leakage, and highlights actionable business insights around pricing and tenure. Future improvements could include hyperparameter tuning (GridSearchCV/Optuna), SHAP-based explainability, and deployment as a REST API via FastAPI.

---

*Report generated: June 2026 | Alfido Tech Internship — Task 1*
