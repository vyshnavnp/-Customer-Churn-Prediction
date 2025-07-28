#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import joblib
model = joblib.load('logistic_regression_model.pkl')

# Setup
st.set_page_config(layout="wide")
st.title("Telco Customer Churn - EDA Dashboard")

#import saved csv from eda
df = pd.read_csv('cleaned_churn_data.csv')
df.drop(columns=['customerID'],inplace=True)

# Sidebar
st.sidebar.header("Filter Options")
contract_filter = st.sidebar.multiselect("Contract Type", df['Contract'].unique(), default=df['Contract'].unique())
internet_filter = st.sidebar.multiselect("Internet Service", df['InternetService'].unique(), default=df['InternetService'].unique())

df_filtered = df[df['Contract'].isin(contract_filter) & df['InternetService'].isin(internet_filter)]

# Plotting
def plot_count(column, hue=None, title=""):
    fig, ax = plt.subplots()
    sns.countplot(x=column, hue=hue, data=df_filtered, palette='Set2', ax=ax)
    plt.title(title)
    plt.xticks(rotation=30)
    return fig

def plot_box(x, y, title=""):
    fig, ax = plt.subplots()
    sns.boxplot(x=x, y=y, data=df_filtered, palette='Set2', ax=ax)
    plt.title(title)
    return fig

def plot_hist(x, title=""):
    fig, ax = plt.subplots()
    sns.histplot(data=df_filtered, x=x, hue="Churn", multiple="stack", palette='Set2', bins=30, ax=ax)
    plt.title(title)
    return fig

def plot_heatmap():
    fig, ax = plt.subplots()
    sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title("Correlation Heatmap")
    return fig

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Churn Overview", "Categorical Analysis", "Charges & Tenure", "Correlation", "Prediction"])

with tab1:
    st.subheader("Churn Distribution")
    st.pyplot(plot_count("Churn", title="Churn Count"))

with tab2:
    st.subheader("Contract Type vs Churn")
    st.pyplot(plot_count("Contract", "Churn", "Churn by Contract Type"))

    st.subheader("Internet Service vs Churn")
    st.pyplot(plot_count("InternetService", "Churn", "Churn by Internet Service"))

    st.subheader("Tech Support vs Churn")
    st.pyplot(plot_count("TechSupport", "Churn", "Churn by Tech Support"))

    st.subheader("Senior Citizen vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x='SeniorCitizen', hue='Churn', data=df_filtered, palette='Set2', ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Senior', 'Senior'])
    plt.title("Churn by Senior Citizen Status")
    st.pyplot(fig)

with tab3:
    st.subheader("Monthly Charges vs Churn")
    st.pyplot(plot_box("Churn", "MonthlyCharges", "Monthly Charges by Churn"))

    st.subheader("Total Charges vs Churn")
    st.pyplot(plot_box("Churn", "TotalCharges", "Total Charges by Churn"))

    st.subheader("Tenure Distribution")
    st.pyplot(plot_hist("tenure", "Tenure by Churn"))

with tab4:
    st.subheader("Correlation Heatmap")
    st.pyplot(plot_heatmap())

with tab5:
    st.subheader("🔮 Predict Customer Churn")

    st.markdown("Fill out the form below to predict whether a customer is likely to churn.")

    with st.form("churn_form"):
        gender = st.selectbox("Gender", ['Female', 'Male'])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ['Yes', 'No'])
        dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
        tenure = st.slider("Tenure (in months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment_method = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # 🚀 Convert form inputs to the exact types your model expects
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': bool(senior),  # assuming senior is 0 or 1
            'Partner': True if partner == 'Yes' else False,
            'Dependents': True if dependents == 'Yes' else False,
            'tenure': int(tenure),
            'PhoneService': True if phone_service == 'Yes' else False,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': True if paperless_billing == 'Yes' else False,
            'PaymentMethod': payment_method,
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges)
        }])

        # ⚙️ Prediction using trained model
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # 💬 Display result
        st.markdown("### 🧾 Prediction Result:")
        st.success(f"Customer is likely to **{'churn' if prediction == 1 else 'stay'}**.")
        st.info(f"Predicted Churn Probability: **{probability:.2%}**")

