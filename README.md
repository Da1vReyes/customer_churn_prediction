# Customer Churn Prediction

> Binary classification project that predicts whether a telecom customer is likely to churn.

**Note:** This repository is a rebuild of the Customer Churn Prediction project I developed during my Machine Learning internship at NexAfrica. I lost the original code, so I recreated the full workflow from scratch using the public Telco Customer Churn dataset, following the same steps: data preparation, EDA, feature engineering, classification models, class imbalance handling, optimization and interpretation.

## Project Overview
The goal of this project is to build a Machine Learning model that predicts whether a telecom customer is likely to leave the company (churn) or stay. It simulates a real-world business problem common in telecommunications, banking, insurance, subscription services and e-commerce.

- **Problem type:** supervised learning, binary classification
- **Target variable:** `Churn` (Yes / No)

## Business Problem
Acquiring a new customer is generally more expensive than retaining an existing one, so losing customers directly hurts recurring revenue. By predicting which customers are likely to churn, the company can intervene early with retention offers, contract improvements or proactive support.

> The objective of this project is to develop a Machine Learning classification model that predicts whether a customer is likely to churn based on demographic information, account details, subscription information, service usage, contract type, tenure, and billing information.

## Dataset
- **Source:** Telco Customer Churn (IBM sample data, also available on Kaggle)
- **Size:** 7,043 customers, 21 columns (19 features, 1 identifier, 1 target)
- **Raw data:** `data/raw_data/telco_churn.csv`
- **Cleaned data:** `data/processed_data/telco_churn_clean.csv`
- **Data dictionary:** [`docs/data_dictionary.md`](docs/data_dictionary.md)

### Data quality report
| Check | Result | Action |
|---|---|---|
| Missing values (`NaN`) | None | None needed |
| Blank strings in `TotalCharges` | 11 rows, all with `tenure = 0` (new customers) | Converted column to numeric, blanks set to `0` |
| Duplicate rows / customers | 0 | None needed |
| Inconsistent categories | Redundant `No internet service` / `No phone service` values; `SeniorCitizen` coded as 0/1 | Merged into `No`; converted to `No/Yes` |

### Target variable distribution
| Churn | Customers | Percentage |
|---|---|---|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is moderately imbalanced (about 2.8 : 1), so accuracy alone is not a reliable metric. Models are evaluated with Precision, Recall, F1-score and ROC-AUC.

## Tools and Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, imbalanced-learn, Jupyter Notebook.

## Project Workflow
Business Understanding → Data Collection → Data Cleaning → EDA → Feature Engineering → Model Training → Model Evaluation → Model Optimization → Feature Importance → Business Recommendations

## Exploratory Data Analysis
The EDA (section 5 of the notebook) includes 8+ visualizations saved in [`visuals/`](visuals/). Highlights:

![Churn analysis](visuals/churn_analysis.png)

### Key insights so far
1. **Contract type is the strongest driver:** month-to-month customers churn 42.7%, versus 11.3% (one year) and 2.8% (two year).
2. **New customers churn the most:** 47.4% churn in the first year, falling steadily with tenure.
3. **Higher bills, more churn:** customers paying $70-$90 per month churn about 38%, versus about 11% under $35.
4. **Electronic check users churn 45.3%**, about three times more than customers on automatic payments.
5. **Fiber optic customers churn 41.9%**, versus 19.0% for DSL and 7.4% without internet.
6. **Security and tech support services protect against churn** (about 15% vs 31% without them).
7. **Senior citizens** (41.7%) and customers without partner or dependents churn more; **gender has no effect**.

### Feature engineering
Seven features were created (`tenure_group`, `num_services`, `num_protection_services`, `has_streaming`, `spending_category`, `auto_payment`, `avg_monthly_spend`), `customerID`, `gender` and `PhoneService` were removed, and categorical variables were encoded. Details in [`docs/feature_engineering.md`](docs/feature_engineering.md). The machine-learning-ready dataset is `data/processed_data/telco_churn_ml_ready.csv` (7,043 rows, 31 features + target).

## Machine Learning Models
_Work in progress._

## Evaluation Metrics
_Work in progress._

## Key Findings
_Work in progress._

## Business Recommendations
_Work in progress._

## How to Run the Project
```bash
git clone <repo-url>
cd Customer-Churn-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

## Project Structure
```
Customer-Churn-Prediction/
├── data/
│   ├── raw_data/
│   └── processed_data/
├── docs/
│   ├── data_dictionary.md
│   └── feature_engineering.md
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/churn/
│   ├── config.py
│   ├── data/            # loader.py, cleaning.py
│   ├── features/        # engineering.py, preprocessing.py
│   └── visualization/   # eda.py
├── visuals/
├── README.md
└── requirements.txt
```
