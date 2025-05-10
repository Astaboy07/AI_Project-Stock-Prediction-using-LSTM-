import tkinter as tk
from tkinter import messagebox
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

def predict():
    company_input = entry.get().strip().lower()
    symbol = company_symbol_map.get(company_input)

    if not symbol:
        messagebox.showwarning("Warning", "Company not found in database.")
        return

    try:
        df = fetch_stock_data(symbol)
        current_price = df.iloc[-1].values[0]

        X, y, scaler = preprocess_data(df)
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        last_60 = df[-60:].values
        last_60_scaled = scaler.transform(last_60)
        X_test = np.array([last_60_scaled])
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        result_var.set(f"Current Price: ${current_price:.2f}\nPredicted Price: ${predicted_price:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI layout
root = tk.Tk()
root.title("Stock Price Predictor with LSTM")

tk.Label(root, text="Enter Company Name:").pack(pady=5)
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

tk.Button(root, text="Predict Price", command=predict).pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Arial", 12), justify="left").pack(pady=10)

root.mainloop()