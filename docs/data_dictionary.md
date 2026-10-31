# Data Dictionary

Source: Telco Customer Churn dataset (IBM sample data, also available on Kaggle). One row per customer, 7,043 rows and 21 columns.

| Feature | Description | Type |
|---|---|---|
| customerID | Unique customer identifier | Identifier (not used as a feature) |
| gender | Customer gender (Female / Male) | Categorical |
| SeniorCitizen | Whether the customer is a senior citizen (Yes / No) | Categorical |
| Partner | Whether the customer has a partner | Categorical |
| Dependents | Whether the customer has dependents | Categorical |
| tenure | Number of months the customer has stayed with the company | Numerical |
| PhoneService | Whether the customer has phone service | Categorical |
| MultipleLines | Whether the customer has multiple phone lines | Categorical |
| InternetService | Internet provider type (DSL / Fiber optic / No) | Categorical |
| OnlineSecurity | Online security add-on | Categorical |
| OnlineBackup | Online backup add-on | Categorical |
| DeviceProtection | Device protection add-on | Categorical |
| TechSupport | Tech support add-on | Categorical |
| StreamingTV | Streaming TV add-on | Categorical |
| StreamingMovies | Streaming movies add-on | Categorical |
| Contract | Contract type (Month-to-month / One year / Two year) | Categorical |
| PaperlessBilling | Whether the customer uses paperless billing | Categorical |
| PaymentMethod | Payment method (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)) | Categorical |
| MonthlyCharges | Amount charged to the customer each month | Numerical |
| TotalCharges | Total amount charged over the customer's lifetime | Numerical |
| **Churn** | Whether the customer left the company (Yes / No) | **Target** |

## Notes on the cleaned dataset

- `TotalCharges` is stored as text in the raw file (11 rows contain a blank string). Those 11 customers have `tenure = 0`, so they were converted to numeric and set to `0`.
- `SeniorCitizen` was converted from `0/1` to `No/Yes` to match the other binary columns.
- `No internet service` (in the six add-on columns) and `No phone service` (in `MultipleLines`) were merged into `No`. That information is still available in `InternetService` and `PhoneService`.
