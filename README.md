# Customer Churn

Customer Churn Prediction- Exploratory Data Analysis
Project Overview

This project analyzes customer churn behavior using Kaggle's "Telco Customer Churn" dataset by Blastchar.
The goal is to identify key factors that contribute to customer churn and build a predictive model that can flag at-risk customers with strong accuracy and interpretability.

---
## Business Objective
Customer churn is a major challenge for subscription-based companies.  
This project aims to:
- Understand which contract, billing, and service features most influence churn.  
- Build machine learning models that can predict churn probability for each customer.  
- Provide actionable insights for retention strategies (e.g., targeting month-to-month and electronic check users).

---
## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn) 
- Matplotlib and Seaborn for visualization  
- SMOTE for class imbalance correction  
- RandomizedSearchCV for hyperparameter tuning  
- Precision–Recall threshold optimization for business tradeoff analysis  
- Jupyter Notebook for development and presentation  
---
## Explore the Notebook
For those interested in the code and step-by-step analysis, see
notebooks/EDA and notebooks/Churn_Modeling.ipynb.


---
## Data Preparation
- Data loaded from `data/data_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- No missing values in core features
- `TotalCharges` converted to numeric type and imputed where needed  
- Dropped `customerID` (non-predictive unique key)  
- Converted `Churn` to binary format (Yes=1, No=0)  
---

### Exploratory Data Analysis Highlights
- 26.5% of customers churned (class imbalance present).  
- Class imbalance noted (approximately 3:1 ratio), to be addressed during modeling.
![Alt text](assets/Chrun Distribution.png)


---
## Key Drivers of Churn
- Contract Type: 42.7% churn among month-to-month customers vs only 2.8% for 2-year contracts	Long-term contracts reduce churn risk dramatically.
- Internet Service: 41.9% churn for Fiber optic vs 19% for DSL	Fiber customers churn more, possibly due to pricing or competition.
- Payment Method: Highest churn among Electronic check users	Manual payment types correlate with higher churn.
- Online Security/Tech Support:	Customers without these add-ons churned 2–3× more	Bundled service options improve retention.
- Paperless Billing: Paperless users appear to churn more, but mostly because they are month-to-month and electronic check users 
- Correlation, not causation — confirmed via cross-tab analysis.
---
## Statistical Tests
- Chi-Square tests confirmed strong relationships between churn and categorical variables (Contract, PaymentMethod, OnlineSecurity).
- T-Tests revealed significant differences in MonthlyCharges and Tenure between churned and retained customers.
---

## Repository Structure
```
Customer-Churn/
│
├── data/
│   ├── data_raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── data_processed/
│
├── notebooks/
│   ├── Churn_EDA.ipynb
│   ├── Churn_Modeling.ipynb
│
├── models/
│   ├── xgb_churn_tuned.joblib
│   ├── feature_columns.joblib
│   └── serving_config.json
│
├── README.md
│
└── requirements.txt
```

---
### Baseline Models
Three initial classifiers were trained and compared:
- Logistic Regression  
- Random Forest  
- XGBoost  

All models were evaluated using ROC-AUC, Precision, Recall, and F1-score.  
Initial (pre-SMOTE) results:
| Model | ROC-AUC |
|--------|----------|
| Random Forest | **0.830** |
| XGBoost | 0.822 |
| Logistic Regression | 0.803 |

---
## Handling Class Imbalance
The dataset was balanced using SMOTE (Synthetic Minority Oversampling Technique) to improve recall for churners.

After applying SMOTE:
| Model | ROC-AUC |
|--------|----------|
| XGBoost | **0.830** |
| Logistic Regression | 0.823 |
| Random Forest | 0.820 |


**Insight:**  
SMOTE improved recall-oriented models slightly, confirming its benefit for minority class learning.

---

## Model Optimization

### XGBoost Hyperparameter Tuning
XGBoost was tuned using RandomizedSearchCV (30 iterations, 3-fold CV) across a broad parameter grid.  
The best model achieved ROC-AUC roughly 0.84 on the test set with improved precision–recall balance.

**Key tuned parameters:**
- `n_estimators`: 300–400  
- `max_depth`: 4–6  
- `learning_rate`: 0.05–0.1  
- `subsample`: 0.8–1.0  
- `colsample_bytree`: 0.8–1.0  

---

## Threshold Optimization
The default 0.5 cutoff was adjusted to improve recall of churners.  
Two strategies were tested:
- Maximize F1 score
- Target recall ≥ 0.80 while maintaining acceptable precision  

Resulting metrics show a substantial gain in recall (identifying more churners) with minimal loss of precision — providing a more business-relevant model.

---

## Feature Importance
Top 10 drivers of churn (tuned XGBoost):

1. Contract_Month-to-month  
2. Tenure  
3. PaymentMethod_Electronic check  
4. OnlineSecurity_No  
5. TechSupport_No  
6. InternetService_Fiber optic  
7. TotalCharges  
8. MonthlyCharges  
9. PaperlessBilling_Yes  
10. StreamingTV_Yes  

---
## Final Model Performance
| Metric | Value |
|---------|--------|
| ROC-AUC | **0.84** |
| Recall (Churners) | ~0.81 |
| Precision (Churners) | ~0.70 |
| F1 Score | ~0.75 |

The tuned XGBoost model shows a balanced ability to identify churners accurately while keeping false positives manageable.

---

## Next Steps
- Hyperparameter tuning for Random Forest and Logistic Regression for comparative robustness  
- SHAP explainability analysis for customer-level insights  
- Deployment via Flask or Streamlit to demonstrate real-time churn scoring  
