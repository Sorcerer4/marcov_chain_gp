import yfinance as yf
import os

def getData(tickers,start,end):
    os.makedirs("data", exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)[["Open", "Close"]]
        df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
        df.to_csv(f"data/{ticker}.csv")

def main():
    #S&P500 TOP 10
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA","BRK-B","GOOG"]

    start = "2015-01-01"
    end = "2024-01-01"
    getData(tickers,start,end)

if __name__ == "__main__":
    main()
