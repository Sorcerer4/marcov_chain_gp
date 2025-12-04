import yfinance as yf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lookupUD = {
            'UUU': 0,
            'UUD': 1,
            'UDU': 2,
            'UDD': 3,
            'DUU': 4,
            'DUD': 5,
            'DDU': 6,
            'DDD': 7
        }

def getData(ticker,start,end, folder):
    os.makedirs(folder, exist_ok=True)

    df = yf.download(ticker, start=start, end=end, auto_adjust=False)[["Open", "Close"]]
    df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
    df.to_csv(f"{folder}/{ticker}.csv")

def readData(ticker,folder):
    #Reads changes and closeprices for each stock and return 2 respective dictionairies

    changes = dict()
    prices = dict()

    df = pd.read_csv(f"{folder}/{ticker}.csv")

        #Convert to numerical values

    df["DailyChange"] = pd.to_numeric(df["DailyChange"],errors= "coerce")
    df["Close"] = pd.to_numeric(df["Close"],errors= "coerce")

    dailyPrices = df["Close"].to_numpy()[3:]

    dailyChanges = df["DailyChange"].to_numpy()[3:]

    changes[ticker] = dailyChanges
    prices[ticker] = dailyPrices
    
    return changes, prices

def makeTrMatrix(dailyChanges):
    #build an 8x8 matrix based on U/D sequences
    for ticker,changes in dailyChanges.items():

        changeUD = []
        for change in changes:
            #convert daily numerical change into U (upp/bulish) or D (down/bearish)
            if change < 0:
                changeUD.append('D')
            else:
                changeUD.append('U')
        
        MUD = np.zeros((8,8))

        #fill matrix with each instance of a 3 U/D combination followed by a 3 U/D combination
        for i in range(4, len(changeUD)-4):
        
            back = ''.join(changeUD[i-4:i-1])
            forward = ''.join(changeUD[i+1:i+4])
    
            MUD[lookupUD[forward],lookupUD[back]] += 1
            MUD[2,1] = 1
        
        #convert count of instances in matrix to probabilities
        for i in range (0,8):
            jSum=0
            for j in range(0,8):
                jSum += MUD[j, i]

            for j in range(0,8):
                MUD[j, i] /= jSum 
        return MUD

def signalMUD(MUD, changes):
    #convert numeric changes to U/D
    changeUD = []
    signals = []
    for change in changes:
        if change < 0:
            changeUD.append('D')
        else:
            changeUD.append('U')
    
    #generate signals
    for i in range(3, len(changeUD)-1):
        
        back = ''.join(changeUD[i-3:i])


        INPUT = np.zeros((8, 1))
        INPUT[lookupUD[back], 0] = 1

        output = (np.matmul(MUD, INPUT))

        up = 0
        down = 0
        #sum probabilities of U or D tomorrow 
        for j in range (0,4):
            up += output[j,0]

        for j in range (4,8):
            down += output[j,0]
        if up > 0.50:
            signals.append(1)
        elif down > 0.50:
            signals.append(-1)
        else:
            signals.append(0)

    return signals

def backtest(signals, changes):
    #initial capital, for allignment
    cash = 10000
    dailyEquity = [10000]

    for i in range(len(changes)-4):
        #buy
        if signals[i] == 1:
            cash *= (1+changes[i])

        #short
        elif signals[i] == -1:
            cash *= (1-(changes[i]))

        dailyEquity.append(cash)

    return dailyEquity

def plotResults(results, buyHold, ticker):

    plt.figure(figsize=(10, 6))
    
    plt.plot(buyHold, label='Buy & Hold')

    plt.plot(results, label="Moneymaker")

    plt.legend()
    plt.title(f"{ticker}")
    plt.xlabel("Days since start")
    plt.ylabel("Equity")
    plt.show()


def main():
    ticker = "QQQ"
    start = "2020-01-01"
    end = "2024-01-01"

    testStart = "2024-09-08"
    testEnd = "2025-09-08"

    getData(ticker,start,end, folder="trainData")
    changes_dict , prices_dict = readData(ticker, folder="trainData")

    getData(ticker, testStart, testEnd, folder="testData")
    changes_dicttest, buyHold = readData(ticker, folder="testData")


    MUD = makeTrMatrix(changes_dict)

    signals = signalMUD(MUD, changes_dicttest[ticker])
    
    results = backtest(signals, changes_dicttest[ticker])

    buyHoldNorm = []

    for price in buyHold[ticker][:-3]:
        buyHoldNorm.append(price*(10000/(buyHold[ticker][0])))

    plotResults(results, buyHoldNorm, ticker)


if __name__ == "__main__":
    main()


