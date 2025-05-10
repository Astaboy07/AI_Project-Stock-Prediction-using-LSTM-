import streamlit as st
from fetch_data import fetch_stock_data
from lstm_model import preprocess_data, build_model
import numpy as np

company_symbol_map = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "intel": "INTC",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "ibm": "IBM"
}

st.title("📈 Stock Price Predictor with LSTM")

company_input = st.text_input("Enter Company Name (e.g., Apple, Tesla, Microsoft):").lower()
symbol = company_symbol_map.get(company_input)

if symbol:
    try:
        df = fetch_stock_data(symbol)
        st.write(f"### Current Price of {company_input.title()}: ${df.iloc[-1].values[0]:.2f}")

        X, y, scaler = preprocess_data(df)
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        last_60 = df[-60:].values
        last_60_scaled = scaler.transform(last_60)
        X_test = np.array([last_60_scaled])
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        st.write(f"### Predicted Next Day Price: ${predicted_price[0][0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    if company_input:
        st.warning("Company not found in the list. Try entering a valid company name.")