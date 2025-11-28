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

        dailyPrices = df["Close"].to_numpy()[3:]



        dailyChanges = df["DailyChange"].to_numpy()[3:]


        changes[ticker] = dailyChanges
        prices[ticker] = dailyPrices
    return changes, prices

def mak_tr_matrix(dailyChanges):
 
    for ticker,changes in dailyChanges.items():

        changeUD = []
        for change in changes:
            if change < 0:
                changeUD.append('D')
            else:
                changeUD.append('U')
        
        MUD = np.zeros((8,8))


        for i in range(4, len(changeUD)-4):
        
            back = ''.join(changeUD[i-4:i-1])
            forward = ''.join(changeUD[i+1:i+4])
    
            MUD[lookupUD[forward],lookupUD[back]] += 1
            MUD[2,1] = 1
        

        for i in range (0,8):
            jsum=0
            for j in range(0,8):
                jsum += MUD[j, i]

            for j in range(0,8):
                MUD[j, i] /= jsum 
        return MUD

def signalMUD(MUD, changes):
    changeUD = []
    signals = [0, 0, 0, 0]
    for change in changes:
        if change < 0:
            changeUD.append('D')
        else:
            changeUD.append('U')
    
    for i in range(4, len(changeUD)-4):
        
        back = ''.join(changeUD[i-4:i-1])


        INPUT = np.zeros((8, 1))
        INPUT[lookupUD[back], 0] = 1

        output = (np.matmul(MUD, INPUT))

        up = 0
        down = 0

        for j in range (0,4):
            up += output[j,0]

        for j in range (4,8):
            down += output[j,0]
        if up > 0.50:
            signals.append(1)
        elif down > 0.55:
            signals.append(-1)
        else:
            signals.append(0)

    signals.append(0)
    signals.append(0)
    signals.append(0)
    return signals

def backtest(signals, changes):
    cash = 10000
    dailyequity = [10000, 10000, 10000, 10000]

    for i in range(0, len(changes)-5):
        x = i + 1
        if signals[i] == 1:
            cash *= (1+changes[x])
            dailyequity.append(cash-0.5)

        elif signals[i] == -1:
            if(1-(changes[x])) < 1:
                cash *= (1-(changes[x]))
                dailyequity.append(cash)
            else:
                cash *= (1-(changes[x])*0.95)
                dailyequity.append(cash)
        else:
            dailyequity.append(cash)

    return dailyequity

def main():
    tickers = ["PG"]
    start = "2020-01-01"
    end = "2024-01-01"

    teststart = "2024-01-02"
    testend = "2025-10-01"

    getData(tickers,start,end, folder="trainData")
    changes_dict , prices_dict = readData(tickers, folder="trainData")

    getData(tickers, teststart, testend, folder="testData")
    changes_dicttest, real = readData(tickers, folder="testData")


    MUD = mak_tr_matrix(changes_dict)

    #Setup plot
    plt.figure(figsize=(10, 6))

    signals = signalMUD(MUD, changes_dicttest[tickers[0]])
    
    results = backtest(signals, changes_dicttest[tickers[0]])
    
    realspy = []

    for price in real[tickers[0]]:
        realspy.append(price*(10000/float(real[tickers[0]][0])))


    plt.plot(realspy, label='Buy & Hold')

    plt.plot(results, label="Moneymaker")

    plt.legend()
    plt.title(f"{tickers[0]}")
    plt.xlabel("Days since start")
    plt.ylabel("Equity")
    plt.show()


if __name__ == "__main__":
    main()


