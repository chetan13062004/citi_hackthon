import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Load the trained models
with open('credit_score_model.pkl', 'rb') as f:
    credit_score_model = pickle.load(f)

with open('loan_risk_model_final.pkl', 'rb') as f:
    loan_risk_model = pickle.load(f)

# Sidebar info
st.sidebar.title("💡 AI-Powered Credit Scoring System")
st.sidebar.header(" About Model")
st.sidebar.success("""
                    This AI-powered Credit Scoring and Loan Risk Prediction system is 
                   designed to evaluate a business’s financial health and creditworthiness. 
                   Built using Random Forest for credit score regression and a Decision Tree 
                   for loan risk classification, it uses synthetic business data to make accurate predictions. 
                   The system also leverages Google’s Gemini AI to provide real-time justifications 
                   for the model’s decisions. """)
st.sidebar.header("Features")
st.sidebar.write("Dual Model Integration:")
st.sidebar.write("Real-Time Credit Report:")
st.sidebar.write("Financial Health Visualization:")
st.sidebar.write("Explainable AI Integration:")

# Input form for user data
st.title("📊 Business Credit Score and Loan Risk Prediction")

business_id = st.number_input("🏢 Business ID", min_value=100, max_value=999, value=101)
annual_revenue = st.number_input("💰 Annual Revenue", min_value=100000, value=4994849)
loan_amount = st.number_input("🏦 Loan Amount", min_value=10000, value=318442)
gst_compliance = st.slider("📈 GST Compliance (%)", min_value=0, max_value=100, value=86)
past_defaults = st.number_input("⚠️ Past Defaults", min_value=0, max_value=10, value=0)
bank_transactions = st.selectbox("💳 Bank Transactions", ["Low Volume", "Stable", "High Volume", "Unstable"])
market_trend = st.selectbox("📉 Market Trend", ["Growth", "Stable", "Declining"])
#additional_feature = st.number_input("📊 Additional Feature", min_value=0, max_value=100, value=50)

# Mappings for classification model
bank_transactions_mapping = {"Low Volume": 0, "Stable": 1, "Unstable": 2, "High Volume": 3}
market_trend_mapping = {"Growth": 0, "Stable": 1, "Declining": 2}

# Encode classification input
classification_input = np.array([[
    business_id,
    annual_revenue,
    loan_amount,
    gst_compliance,
    past_defaults,
    bank_transactions_mapping[bank_transactions],
    market_trend_mapping[market_trend]
]])

# Encode regression input
Regression_input = np.array([[
    business_id,
    annual_revenue,
    loan_amount,
    gst_compliance,
    past_defaults
] + [
    1 if bank_transactions == "Stable" else 0,
    1 if bank_transactions == "High Volume" else 0,
    1 if bank_transactions == "Unstable" else 0
] + [
    1 if market_trend == "Stable" else 0,
    1 if market_trend == "Declining" else 0
]])

def get_gemini_response(input_text):
        model = genai.GenerativeModel('gemini-1.5-pro-002')
        response = model.generate_content(input_text)
        return response.text

# Predict credit score
if st.button("🔍 Predict Credit Score"):
    credit_score = credit_score_model.predict(Regression_input)[0]

    st.success(f"🏆 Predicted Credit Score: {credit_score:.2f}")

    # Visualize revenue vs. loan amount
    fig, ax = plt.subplots()
    ax.bar(["Annual Revenue", "Loan Amount"], [annual_revenue, loan_amount], color=['blue', 'orange'])
    ax.set_ylabel("Amount")
    ax.set_title("💸 Business Financial Overview")
    st.pyplot(fig)

    # Send result and inputs to Gemini API
    input_text = (f"Business with annual revenue of {annual_revenue}, loan amount {loan_amount}, GST compliance {gst_compliance}%, "
                  f"{past_defaults} past defaults, {bank_transactions} bank transactions, and a market trend of {market_trend} "
                  f"has a predicted credit score of {credit_score:.2f}. and your task is to explain the decision of the model")

   

    explanation = get_gemini_response(input_text)
    st.write("### 🤖 AI Analysis")
    st.write(explanation)

# Predict loan risk
if st.button("⚠️ Predict Loan Risk"):
    loan_risk = loan_risk_model.predict(classification_input)[0]

    risk_labels = {0: "Minimal Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
    st.success(f"⚠️ Predicted Loan Risk: {risk_labels.get(loan_risk, 'Unknown Risk')}")

    # Visualize loan risk as a bar chart
    risk_levels = ["Minimal Risk", "Low Risk", "Medium Risk", "High Risk"]
    risk_values = [1 if risk_labels.get(loan_risk) == level else 0 for level in risk_levels]

    fig, ax = plt.subplots()
    ax.bar(risk_levels, risk_values, color=['green', 'yellow', 'orange', 'red'])
    ax.set_ylabel("Risk Level")
    ax.set_title("📉 Loan Risk Classification")
    st.pyplot(fig)

    input_text = (f"Business with revenue {annual_revenue}, loan amount {loan_amount}, and {past_defaults} past defaults "
                  f"is classified as having {risk_labels.get(loan_risk, 'Unknown Risk')}. Explain the classification.")

    explanation = get_gemini_response(input_text)
    st.write("### 🧠 AI Explanation")
    st.write(explanation)
