import yfinance as yf
import os
import pandas as pd
import numpy as np
from numpy.linalg import matrix_power

def getData(tickers,start,end):
    os.makedirs("data", exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)[["Open", "Close"]]
        df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
        df.to_csv(f"data/{ticker}.csv")

def readData(tickers):
    d = dict()
    for ticker in tickers:
        df = pd.read_csv(f"data/{ticker}.csv")
        dailyChanges = df["DailyChange"].to_numpy()[2:]
        d[ticker] = dailyChanges
    return d

def countDiscreteChange(dailyChanges):
    bins = np.arange(-0.1, 0.105, 0.005)
    bins = np.insert(bins, 0, -np.inf)
    bins = np.append(bins, np.inf)

    for changes in dailyChanges.values():
        counts,edges = np.histogram(changes,bins)
        print("Counts per interval:", counts)

def mak_tr_matrix(dailyChanges):

    m,n = 2,2

    for name,changes in dailyChanges.items():
        M = np.zeros((m,n))
        count_bull = count_bear = 0

        print(name)

        for i in range(len(changes)-1):
            if changes[i] < 0 and changes[i+1] < 0:
                M[0,0] += 1
                count_bear +=1
            elif changes[i] < 0 and changes[i+1] > 0:
                M[1,0] += 1
                count_bear += 1
            elif changes[i] > 0 and changes[i+1] < 0:
                M[0,1] += 1
                count_bull += 1
            elif changes[i] > 0 and changes[i+1] > 0:
                M[1,1] += 1
                count_bull += 1


        M[0,0], M[1,0] = M[0,0]/count_bear, M[1,0]/count_bear
        M[0, 1], M[1, 1] = M[0, 1] / count_bull, M[1, 1] / count_bull

        for i in range(12):
            print((matrix_power(M,i)))




def main():
    #S&P500 TOP 10
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA","BRK-B","GOOG"]

    start = "2023-01-01"
    end = "2024-01-01"

    getData(tickers,start,end)
    dailychanges = readData(tickers)
    mak_tr_matrix(dailychanges)
    #countDiscreteChange(dailychanges)



if __name__ == "__main__":
    main()
