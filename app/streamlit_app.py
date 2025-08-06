import streamlit as st
import numpy as np
import pickle
import joblib

# Load trained model
model = joblib.load('models/final_xgb_model.pkl')
st.title("E-commerce Customer Churn Prediction")

st.sidebar.header("User Input Features")

# Input fields for user
Tenure = st.sidebar.number_input("Tenure", min_value=0.0, step=0.1)
WarehouseToHome = st.sidebar.number_input("Warehouse To Home", min_value=0.0, step=0.1)
NumberOfDeviceRegistered = st.sidebar.number_input("Number Of Devices Registered", min_value=0)
SatisfactionScore = st.sidebar.slider("Satisfaction Score", 0, 5, 3)
NumberOfAddress = st.sidebar.number_input("Number Of Addresses", min_value=0)
Complain = st.sidebar.selectbox("Complain", [0, 1])
DaySinceLastOrder = st.sidebar.number_input("Days Since Last Order", min_value=0)
CashbackAmount = st.sidebar.number_input("Cashback Amount", min_value=0.0, step=0.01)

PreferedOrderCat = st.sidebar.selectbox("Preferred Order Category", 
    ['Grocery', 'Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others'])

MaritalStatus = st.sidebar.selectbox("Marital Status", ['Married', 'Single'])

# One-hot encode PreferedOrderCat
order_cat = {
    'PreferedOrderCat_Grocery': 0,
    'PreferedOrderCat_Laptop & Accessory': 0,
    'PreferedOrderCat_Mobile': 0,
    'PreferedOrderCat_Mobile Phone': 0,
    'PreferedOrderCat_Others': 0
}
order_cat[f'PreferedOrderCat_{PreferedOrderCat}'] = 1

# One-hot encode MaritalStatus
marital_status = {
    'MaritalStatus_Married': 0,
    'MaritalStatus_Single': 0
}
marital_status[f'MaritalStatus_{MaritalStatus}'] = 1

# Create input array
input_data = np.array([[
    Tenure, WarehouseToHome, NumberOfDeviceRegistered, SatisfactionScore,
    NumberOfAddress, Complain, DaySinceLastOrder, CashbackAmount,
    *order_cat.values(), *marital_status.values()
]])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    st.write("Prediction:", "Churn" if prediction == 1 else "Not Churn")
