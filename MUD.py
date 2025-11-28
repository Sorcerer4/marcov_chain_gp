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
    signals = [0, 0, 0, 0]
    for change in changes:
        if change < 0:
            changeUD.append('D')
        else:
            changeUD.append('U')
    
    #generate signals
    for i in range(4, len(changeUD)-4):
        
        back = ''.join(changeUD[i-4:i-1])


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
        elif down > 0.55:
            signals.append(-1)
        else:
            signals.append(0)

    #for allignment
    signals.append(0)
    signals.append(0)
    signals.append(0)
    return signals

def backtest(signals, changes):
    #initial capital, for allignment
    cash = 10000
    dailyEquity = [10000, 10000, 10000, 10000]

    for i in range(0, len(changes)-5):
        x = i + 1
        #buy
        if signals[i] == 1:
            cash *= (1+changes[x])
            dailyEquity.append(cash-0.5)

        #short
        elif signals[i] == -1:
            #and unsuccesful
            if(1-(changes[x])) < 1:
                cash *= (1-(changes[x]))
                dailyEquity.append(cash)
            #and succesful
            else:
                cash *= (1-(changes[x])*0.95)
                dailyEquity.append(cash)
        #no trade today
        else:
            dailyEquity.append(cash)

    return dailyEquity

def main():
    tickers = ["PG"]
    start = "2020-01-01"
    end = "2024-01-01"

    testStart = "2024-01-02"
    testEnd = "2025-10-02"

    getData(tickers,start,end, folder="trainData")
    changes_dict , prices_dict = readData(tickers, folder="trainData")

    getData(tickers, testStart, testEnd, folder="testData")
    changes_dicttest, real = readData(tickers, folder="testData")


    MUD = mak_tr_matrix(changes_dict)

    #Setup plot
    plt.figure(figsize=(10, 6))

    signals = signalMUD(MUD, changes_dicttest[tickers[0]])
    
    results = backtest(signals, changes_dicttest[tickers[0]])
    
    buyHold = []

    for price in real[tickers[0]]:
        buyHold.append(price*(10000/float(real[tickers[0]][0])))


    plt.plot(buyHold, label='Buy & Hold')

    plt.plot(results, label="Moneymaker")

    plt.legend()
    plt.title(f"{tickers[0]}")
    plt.xlabel("Days since start")
    plt.ylabel("Equity")
    plt.show()


if __name__ == "__main__":
    main()


