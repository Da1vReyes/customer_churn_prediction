# Feature Engineering and Preprocessing

Implemented in `src/churn/features/engineering.py` and `src/churn/features/preprocessing.py`.

## Engineered features

| Feature | How it was created | Why it may help predict churn |
|---|---|---|
| `tenure_group` | `tenure` binned into 0-12, 13-24, 25-48 and 49-72 months | Churn follows a lifecycle: 47.4% in the first year vs 9.5% after four years |
| `num_services` | Count of phone, multiple lines, internet and the six add-ons | Measures how tied the customer is to the company |
| `num_protection_services` | Count of `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | Churn falls from 38.9% (1 service) to 5.3% (all 4) |
| `has_streaming` | `Yes` if `StreamingTV` or `StreamingMovies` is `Yes` | Single indicator for entertainment services |
| `spending_category` | `MonthlyCharges` in Low (up to $35), Medium ($35-$70), High (above $70) | High spenders churn 35.4% vs 10.9% for low spenders |
| `auto_payment` | `Yes` if payment method is bank transfer or credit card (automatic) | 16.0% churn with automatic payment vs 34.7% without |
| `avg_monthly_spend` | `TotalCharges / tenure` (current `MonthlyCharges` when tenure is 0) | Compares the current bill with the historical one |

## Feature selection

| Decision | Features | Reason |
|---|---|---|
| Dropped | `customerID` | Identifier, not a predictor |
| Dropped | `gender`, `PhoneService` | Chi-square test: no relation with churn (p = 0.49 and p = 0.34) |
| Kept | `TotalCharges` | Correlated with `tenure` (0.83) but harmless for tree models; revisited in Week 3 for Logistic Regression |
| Kept | All other original and engineered features | Statistically related to churn (p < 0.05) |

## Preprocessing

- **Encoding:** Yes/No columns and the target become 1/0. `InternetService`, `Contract`, `PaymentMethod`, `tenure_group` and `spending_category` are one-hot encoded (first level dropped).
- **Scaling:** `scale_numeric()` applies `StandardScaler` to the numeric features. It is fitted on the training set only (after the Week 3 split) to avoid data leakage, so the saved dataset is left unscaled.
- **Outliers:** none found with the IQR rule, so no rows were removed.

## Output datasets

| File | Content |
|---|---|
| `data/processed_data/telco_churn_features.csv` | Cleaned data plus the 7 engineered features (readable categories) |
| `data/processed_data/telco_churn_ml_ready.csv` | Selected features, fully numeric, target `Churn` as 1/0 (7,043 rows x 32 columns) |
