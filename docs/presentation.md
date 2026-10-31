# Customer Churn Prediction: Presentation Outline (5-10 minutes)

**1. The business problem (1 slide).** Losing customers hurts recurring revenue and acquiring new ones costs more than keeping them. Goal: predict who is likely to churn to act early.

**2. The dataset (1 slide).** Telco Customer Churn: 7,043 customers, 19 features, 26.5% churn (about 2.8 : 1 imbalance).

**3. Data preparation (1 slide).** 11 blank `TotalCharges` (new customers) set to 0, no duplicates, redundant categories merged, `SeniorCitizen` recoded.

**4. Key EDA findings (1-2 slides).** `visuals/churn_analysis.png`: month-to-month 42.7% vs two-year 2.8%; first-year customers 47.4%; electronic check 45.3%; fiber optic 41.9%; security/support services protect.

**5. Feature engineering (1 slide).** 7 new features (`tenure_group`, `num_services`, `num_protection_services`, `has_streaming`, `spending_category`, `auto_payment`, `avg_monthly_spend`); `gender` and `PhoneService` removed.

**6. Models tested (1 slide).** Logistic Regression (baseline), Decision Tree, Random Forest, Gradient Boosting (XGBoost fallback).

**7. Model comparison (1 slide).** `visuals/model_performance.png`: all around 79% accuracy but only about 50% recall, so accuracy is misleading.

**8. Imbalance and optimization (1 slide).** `visuals/imbalance_strategies.png`: class weights raise recall from 50% to about 80%; cross-validation confirms consistency; tuning fixes Random Forest overfitting.

**9. Final model (1 slide).** Tuned Random Forest: recall 0.78, precision 0.55, F1 0.64, ROC-AUC 0.845 (`visuals/final_model_comparison.png`).

**10. Feature importance (1 slide).** `visuals/feature_importance.png`: contract, fiber optic, tenure, payment method, protection services.

**11. Business recommendations (1 slide).** Longer contracts, onboarding, review fiber offer, automatic payment, add-on trials, risk-ranked campaigns (`visuals/risk_deciles.png`: top 20% = 50% of churners).

**12. Limitations and next steps (1 slide).** Not causal, 55% precision, missed churners, threshold tuning, SHAP, behavioral data.
