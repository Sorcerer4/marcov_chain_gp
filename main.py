import yfinance as yf
import os
import pandas as pd
import numpy as np

from Backtester import backtest

class stock:
    def __init__(self,ticker,training_start, training_end, strategy_fn, trainingperiod = 12, testperiod = 6):
        self.ticker = ticker
        self.start = training_start
        self.end = training_end
        self.trainingperiod = trainingperiod
        self.testperiod = testperiod
        self.strategyfunction = strategy_fn

        self.performance_log = self.doCrossValidation(0,0.005,False)
        self.avgPerformance = np.mean(self.performance_log)
        self.volPerformance = np.std(self.performance_log)

    def doCrossValidation(self, cash, fee,allow_shorting):
        starting_Dates = pd.date_range(start=self.start, end=self.end, freq='MS').date  # MS = Month Start

        performance_log = []
        for start in starting_Dates:
            end = start + pd.DateOffset(months=self.trainingperiod)
            testing_start = end + pd.DateOffset(months=1)
            testing_end = testing_start + pd.DateOffset(months=self.testperiod)

            start = start.strftime('%Y-%m-%d')
            end = end.strftime('%Y-%m-%d')
            testing_start = testing_start.strftime('%Y-%m-%d')
            testing_end = testing_end.strftime('%Y-%m-%d')

            self.getData(self.ticker, start, end, folder="trainData")
            self.getData(self.ticker, testing_start, testing_end, folder="testData")

            train_changes, train_prices = self.readData(self.ticker, folder="trainData")
            test_changes, test_prices = self.readData(self.ticker, folder="testData")

            marcovMatrix = self.mk_tr_matrix(train_changes)

            result = backtest(self.ticker, test_prices, marcovMatrix, self.strategyfunction, fee, cash, allow_shorting)
            profit = (result[-1] - result[0]) / result[0]
            performance_log.append(profit)

        return performance_log


    def mk_tr_matrix(self,changes):

        m, n = 2, 2
        count_bull = count_bear = 0

        M = np.zeros((m, n))

        for i in range(len(changes) - 1):
            if changes[i] < 0 and changes[i + 1] < 0:
                M[0, 0] += 1
                count_bear += 1
            elif changes[i] < 0 and changes[i + 1] > 0:
                M[1, 0] += 1
                count_bear += 1
            elif changes[i] > 0 and changes[i + 1] < 0:
                M[0, 1] += 1
                count_bull += 1
            elif changes[i] > 0 and changes[i + 1] > 0:
                M[1, 1] += 1
                count_bull += 1
        M[0, 0], M[1, 0] = M[0, 0] / count_bear, M[1, 0] / count_bear
        M[0, 1], M[1, 1] = M[0, 1] / count_bull, M[1, 1] / count_bull

        return M

    def getData(self, name, start, end, folder):
        os.makedirs(folder, exist_ok=True)
        df = yf.download(name, start=start, end=end, auto_adjust=True)[["Open", "Close"]]
        df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
        df.to_csv(f"{folder}/{name}.csv")

    def readData(self,name, folder):
        # Reads changes and closeprices for each stock and return 2 respective dictionairies

        df = pd.read_csv(f"{folder}/{name}.csv")

        # Convert to numerical values

        df["DailyChange"] = pd.to_numeric(df["DailyChange"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        dailyChanges = df["DailyChange"].to_numpy()[3:]
        dailyPrices = df["Close"].to_numpy()[3:]

        return dailyChanges, dailyPrices

def marcovTrade(ticker, position, historical_prices, day, cash, matrix):

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


def main():

    ticker = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "GOOG"]
    training_start = "2020-01-01"
    training_end = "2022-01-01"

    SD = []
    AVG = []

    for ticker in ticker:
        s = stock(ticker, training_start, training_end, marcovTrade)
        SD.append(s.volPerformance)
        AVG.append(s.avgPerformance)

    print(SD)
    print(AVG)

if __name__ == "__main__":
    main()
