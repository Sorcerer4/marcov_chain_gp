import yfinance as yf
import os
import pandas as pd
import numpy as np
import random
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt


from Backtester import backtest

def getData(tickers,start,end, folder):
    os.makedirs(folder, exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)[["Open", "Close"]]
        df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
        df.to_csv(f"{folder}/{ticker}.csv")

def readData(tickers,folder):
    #Reads changes and closeprices for each stock and return 2 respective dictionairies

    changes = dict()
    prices = dict()
    for ticker in tickers:
        df = pd.read_csv(f"{folder}/{ticker}.csv")

        #Convert to numerical values

        df["DailyChange"] = pd.to_numeric(df["DailyChange"],errors= "coerce")
        df["Close"] = pd.to_numeric(df["Close"],errors= "coerce")



        dailyChanges = df["DailyChange"].to_numpy()[3:]
        dailyPrices = df["Close"].to_numpy()[3:]


        changes[ticker] = dailyChanges
        prices[ticker] = dailyPrices
    return changes,prices

def mak_tr_matrix(dailyChanges):

    dictMatrices = dict()
    m,n = 2,2

    for ticker,changes in dailyChanges.items():
        M = np.zeros((m,n))
        count_bull = count_bear = 0

        print(ticker)

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

        dictMatrices[ticker] = M

        for i in range(12):
            print((matrix_power(M,i)))

    return dictMatrices

def marcovTrade(ticker, position, historical_prices, day, cash, matrix):
    #--------RANDOMTRADE STRATEGY PLACEHOLDER--------

    ##randomNumb = random.randint(-5,5)

    target_position = 0

    changeToday = (historical_prices[day]-historical_prices[day-1])/historical_prices[day-1]

    vector = [[0],[0]]

    if changeToday < 0:
        vector[0].append(0)
        vector[1].append(1)
    else:
        vector[0].append(1)
        vector[1].append(0)

    probVector = np.matmul(matrix, vector)

    if probVector[0][0] > 0.5:
        target_position = position - 10
    else:
        target_position = position + 10



    return target_position

    #hej

def main():
    #S&P500 TOP 10

    tickers = ["NVDA","GOOG"]
    start = "2014-01-01"
    end = "2019-01-01"

    getData(tickers,start,end, folder="testData")
    changes_dict , prices_dict = readData(tickers, folder="testData")

    dictMatrices = mak_tr_matrix(changes_dict)

    #Setup plot
    plt.figure(figsize=(10, 6))

    for ticker, prices in prices_dict.items():
        #mak_tr_matrix(daily_changes)
        print(ticker, prices)

        results = backtest(ticker, prices,marcovTrade,0.005,10000,False, dictMatrices[ticker])



        plt.plot(results, label=ticker)

    #Plotting
    plt.legend()
    plt.title("Results Strategy)")
    plt.xlabel("Days")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    main()
