# Customer Churn Prediction & Retention Risk Scoring System

## Overview
This project predicts which telecom customers are likely to leave the service (churn).  
The goal is to allow companies to proactively retain customers before revenue loss occurs.

Instead of optimizing only for accuracy, the model focuses on identifying high-risk customers early.

---

## Business Objective
Customer acquisition is significantly more expensive than retention.  
Missing a churner leads to direct revenue loss, so the model prioritizes **recall for churn customers** over raw accuracy.

The system acts as a **customer risk scoring tool** rather than a simple classifier.

---

## Dataset
IBM Telco Customer Churn Dataset  
- 7043 customers
- 21 features
- Includes demographics, services, billing, and contract information

---

## Key Insights (EDA)
- New customers churn more frequently than long-term customers
- Month-to-month contracts have the highest churn
- Electronic check users churn more than auto-pay users
- Customer tenure strongly influences retention

---

## Machine Learning Pipeline
Steps performed:

1. Data cleaning (fixed hidden missing values in `TotalCharges`)
2. Feature preprocessing using `ColumnTransformer`
3. One-hot encoding for categorical features
4. Feature scaling using `StandardScaler`
5. Logistic Regression classification
6. Class imbalance handling using `class_weight='balanced'`
7. Probability prediction and threshold tuning

---

## Model Performance

Default Threshold (0.5)
- Recall (churn): ~0.78
- Accuracy: ~0.74

Optimized Threshold (0.2)
- Recall (churn): ~0.96
- Captures most potential churners

The model is optimized for **business retention impact**, not just statistical accuracy.

---

## Project Structure

```
customer-churn-ml/
│
├── data/
│   └── telco_churn.csv
│
├── model/
│   └── churn_model.pkl
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── src/
│   └── test_model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```