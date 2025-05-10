import requests
import pandas as pd

API_KEY = "G35HGMTQ1JHXR85M"

def fetch_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError("Invalid Symbol or API limit exceeded")

    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df = df.astype(float)
    df = df.sort_index()
    return df[["4. close"]]