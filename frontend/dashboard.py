import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests  # 👈 Added for API communication

# Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Get API URL from environment variable, default to localhost if not set
# In Docker Compose, we will set this to "http://api_service:80/predict"
API_URL = os.getenv("API_URL", "http://localhost:8080/predict")

# --- CACHING DATA ---
@st.cache_data
def load_data():
    # Make sure this points to your data file
    df = pd.read_csv("data/Customer_data.csv") 
    # Basic cleaning for display
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: Could not load data. Please ensure 'data/Customer_data.xlsx' exists.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🔍 Filter Options")
st.sidebar.write("Use these filters to slice the data for the EDA section.")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=df['PaymentMethod'].unique(),
    default=df['PaymentMethod'].unique()
)

# Apply filters
df_filtered = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter)) &
    (df['PaymentMethod'].isin(payment_filter))
]

# --- MAIN PAGE ---
st.title("📊 Telco Customer Churn Dashboard")
st.markdown(f"**Backend Status:** Sending predictions to `{API_URL}`")

# KPI Metrics
total_customers = len(df_filtered)
churn_rate = df_filtered['Churn'].value_counts(normalize=True).get('Yes', 0)
avg_monthly = df_filtered['MonthlyCharges'].mean()
avg_tenure = df_filtered['tenure'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.1%}")
col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Analysis", "💳 Financial Insights", "🔮 Churn Prediction"])

# --- TAB 1 & 2 (Visualization Code - Kept same as before) ---
with tab1:
    st.subheader("Customer Demographics & Services")
    col_a, col_b = st.columns(2)
    with col_a:
        churn_counts = df_filtered['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn Status', 'Count']
        fig_churn = px.pie(churn_counts, names='Churn Status', values='Count', 
                           title='Overall Churn Distribution', color='Churn Status',
                           color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'})
        st.plotly_chart(fig_churn, use_container_width=True)
    with col_b:
        fig_tenure = px.histogram(df_filtered, x='tenure', color='Churn', barmode='overlay',
                                  title='Customer Tenure Distribution',
                                  color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'})
        st.plotly_chart(fig_tenure, use_container_width=True)

with tab2:
    st.subheader("Financial Factors")
    fig_contract = px.bar(df_filtered.groupby('Contract')['Churn'].value_counts(normalize=True).reset_index(name='Rate'), 
                          x='Contract', y='Rate', color='Churn', title='Churn Rate by Contract')
    st.plotly_chart(fig_contract, use_container_width=True)

# --- TAB 3: CHURN PREDICTION (UPDATED FOR API) ---
with tab3:
    st.subheader("🔮 Real-time Prediction via API")
    
    with st.form("prediction_form"):
        # Section 1: Demographics
        st.markdown("### 👤 Demographics")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        gender = col_p1.selectbox("Gender", ['Female', 'Male'])
        senior = col_p2.selectbox("Senior Citizen", [0, 1], help="0=No, 1=Yes")
        partner = col_p3.selectbox("Has Partner?", ['Yes', 'No'])
        dependents = col_p4.selectbox("Has Dependents?", ['Yes', 'No'])

        # Section 2: Account Info
        st.markdown("### 📝 Account Information")
        col_a1, col_a2, col_a3 = st.columns(3)
        tenure = col_a1.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        contract = col_a2.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        paperless = col_a3.selectbox("Paperless Billing?", ['Yes', 'No'])
            
        col_a4, col_a5, col_a6 = st.columns(3)
        payment = col_a4.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = col_a5.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = col_a6.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

        # Section 3: Services
        st.markdown("### 📡 Services")
        col_s1, col_s2, col_s3 = st.columns(3)
        phone_service = col_s1.selectbox("Phone Service?", ['Yes', 'No'])
        multiple_lines = col_s1.selectbox("Multiple Lines?", ['No', 'Yes', 'No phone service'])
        internet_service = col_s1.selectbox("Internet Service Type", ['DSL', 'Fiber optic', 'No'])
        
        online_security = col_s2.selectbox("Online Security?", ['No', 'Yes', 'No internet service'])
        online_backup = col_s2.selectbox("Online Backup?", ['No', 'Yes', 'No internet service'])
        device_protection = col_s2.selectbox("Device Protection?", ['No', 'Yes', 'No internet service'])
            
        tech_support = col_s3.selectbox("Tech Support?", ['No', 'Yes', 'No internet service'])
        streaming_tv = col_s3.selectbox("Streaming TV?", ['No', 'Yes', 'No internet service'])
        streaming_movies = col_s3.selectbox("Streaming Movies?", ['No', 'Yes', 'No internet service'])

        submit_btn = st.form_submit_button("Predict Churn Risk")

    if submit_btn:
        # 1. Prepare Payload matching schemas.py
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }

        try:
            # 2. Send Request to Docker API
            with st.spinner('Sending data to API...'):
                response = requests.post(API_URL, json=payload)
            
            # 3. Handle Response
            if response.status_code == 200:
                result = response.json()
                prediction = result['churn_prediction']
                probability = result['churn_probability']
                risk = result['risk_level']

                st.divider()
                st.markdown("### 🎯 API Prediction Results")
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    if prediction == 1:
                        st.error("⚠️ Prediction: **CHURN**")
                    else:
                        st.success("✅ Prediction: **STAY**")
                    
                    st.metric("Risk Level", risk)
                    st.metric("Churn Probability", f"{probability:.2%}")

                with col_res2:
                    # Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk %"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "black"},
                            'steps': [
                                {'range': [0, 40], 'color': "green"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}]
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error: Is the Docker container running? (Try http://localhost:8080)")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")