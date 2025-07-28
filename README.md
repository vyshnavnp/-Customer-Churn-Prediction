# 📉 Telco Customer Churn Prediction Dashboard

This project is a complete end-to-end solution for predicting customer churn in a telecom company. 
It combines data preprocessing, model training, evaluation, and deployment using **Streamlit** as an interactive dashboard.

##  Project Overview

- **Problem**: Predict whether a customer is likely to churn based on their account information and service usage.
- **Dataset**: Telco Customer Churn dataset.
- **Goal**: Build a predictive model and provide business insights using interactive visualizations and a prediction tool.

---

##  Features

###  EDA Dashboard

- Contract type vs churn
- Internet service and support features vs churn
- Tenure, Monthly & Total Charges analysis
- Correlation heatmap

###  Prediction Tool

- Fillable form to simulate a new customer
- Real-time churn prediction with churn probability
- Backend model includes full preprocessing pipeline

---

##  Tech Stack

| Component      | Tool/Library              |
|----------------|---------------------------|
| Language       | Python 3.12               |
| Dashboard      | Streamlit                 |
| Data Viz       | Matplotlib, Seaborn       |
| Modeling       | Scikit-learn              |
| Data Handling  | Pandas, NumPy             |
| Model File     | `logistic_regression_model.pkl` |
| Dataset        | `cleaned_churn_data.csv`  |

---

##  Model Development

- Used a pipeline with `ColumnTransformer` for preprocessing:
  - StandardScaler for numerical features
  - OneHotEncoding for categorical features
- Binary columns (e.g., `Partner`, `SeniorCitizen`) were handled properly
- Optimized using GridSearchCV with **Recall** as the scoring metric
- Final model: **Logistic Regression**

---




