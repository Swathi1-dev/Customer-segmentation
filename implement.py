import pandas as pd
import numpy as np
import pickle
import streamlit as st

with open("label_ducation.pkl", "rb") as f:
    le_edu = pickle.load(f)

with open("label_marital.pkl", "rb") as f:
    le_marital = pickle.load(f)

with open("model.keras", "rb") as f:
    model_lr = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

st.title("Customer Segmentation")

Year_Birth = st.number_input("Year", min_value=1950, max_value=2010)
Education = st.selectbox("Education", le_edu.classes_)
Marital_Status = st.selectbox("Marital Status", le_marital.classes_)
Income = st.number_input("Income", min_value=0, max_value=1000000)
Kidhome = st.number_input("Kid Home", min_value=0, max_value=10)
Teenhome = st.number_input("Teen Home", min_value=0, max_value=10)
Dt_Customer = st.date_input("Customer Since")
Recency = st.number_input("Recency", min_value=0, max_value=100)
MntWines = st.number_input("Money Spent on Wines", min_value=0, max_value=100000)
MntFruits = st.number_input("Money Spent on Fruits", min_value=0, max_value=100000)
MntMeatProducts = st.number_input(
    "Money Spent on Meat Products", min_value=0, max_value=100000
)
MntFishProducts = st.number_input(
    "Money Spent on Fish Products", min_value=0, max_value=100000
)

MntSweetProducts = st.number_input(
    "Money Spent on Sweet Products", min_value=0, max_value=100000
)
MntGoldProds = st.number_input(
    "Money Spent on Gold Products", min_value=0, max_value=100000
)
NumDealsPurchases = st.number_input(
    "Number of Deals Purchases", min_value=0, max_value=100
)
NumWebPurchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100)
NumCatalogPurchases = st.number_input(
    "Number of Catalog Purchases", min_value=0, max_value=100
)
NumStorePurchases = st.number_input(
    "Number of Store Purchases", min_value=0, max_value=100
)
NumWebVisitsMonth = st.number_input(
    "Number of Web Visits per Month", min_value=0, max_value=100
)
AcceptedCmp3 = st.selectbox("Accepted Campaign 3", [0, 1])
AcceptedCmp4 = st.selectbox("Accepted Campaign 4", [0, 1])
AcceptedCmp5 = st.selectbox("Accepted Campaign 5", [0, 1])

AcceptedCmp1 = st.selectbox("Accepted Campaign 1", [0, 1])

AcceptedCmp2 = st.selectbox("Accepted Campaign 2", [0, 1])
Complain = st.selectbox("Complain", [0, 1])
Z_CostContact = st.number_input("Cost Contact", min_value=0, max_value=100000)
Z_Revenue = st.number_input("Revenue", min_value=0, max_value=100000)

input_data = np.array(
    [
        [
            Year_Birth,
            le_edu.transform([Education])[0],
            le_marital.transform([Marital_Status])[0],
            Income,
            Kidhome,
            Teenhome,
            Recency,
            MntWines,
            MntFruits,
            MntMeatProducts,
            MntFishProducts,
            MntSweetProducts,
            MntGoldProds,
            NumDealsPurchases,
            NumWebPurchases,
            NumCatalogPurchases,
            NumStorePurchases,
            NumWebVisitsMonth,
            AcceptedCmp3,
            AcceptedCmp4,
            AcceptedCmp5,
            AcceptedCmp1,
            AcceptedCmp2,
            Complain,
            Z_CostContact,
            Z_Revenue,
        ]
    ]
)

if st.button("Predict"):
    input_data_scaled = sc.transform(input_data)
    prediction = model_lr.predict(input_data_scaled)
    st.write(f"Predicted Customer Segment: {prediction[0]}")
else:
    st.write("Please enter the customer details and click Predict.")
