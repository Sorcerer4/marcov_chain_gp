#Projekt: Markovkedjor
#Gruppnummer: 9
#Medlemmar: Luke Bohlin, Aston Ehrling, Victor Jia Gao, Bruno Harju-Jeanty, Arvid Hult

import yfinance as yf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

LOOKUP_UD = {
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

    df = yf.download(ticker, 
                     start=start, 
                     end=end, 
                     auto_adjust=True)[["Open", "Close"]]
    
    df["DailyChange"] = df["Close"].pct_change()
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
    for ticker,changes in dailyChanges.items():

        changeUD = []
        for change in changes:
            if change < 0:
                changeUD.append('D')
            else:
                changeUD.append('U')
        

        MUD = np.zeros((8,8))

        #fill matrix with each instance of a 3 U/D combination followed by a 3 U/D combination
        
        for i in range(4, len(changeUD)-4):
        
            back = ''.join(changeUD[i-3:i])
            forward = ''.join(changeUD[i+1:i+4])
    
            MUD[LOOKUP_UD[forward],LOOKUP_UD[back]] += 1
        
        #convert count of instances in matrix to probabilities
        
        for i in range (0,8):
            jSum=0
            for j in range(0,8):
                jSum += MUD[j, i]

            for j in range(0,8):
                MUD[j, i] /= jSum 
        return MUD

def signalGenerator(MUD, changes):
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
        INPUT[LOOKUP_UD[back], 0] = 1

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

def plotResultsLive(results, buyHold, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Current stock: {ticker}"); ax.set_xlabel("Days"); ax.set_ylabel("Equity")
    ax.set_xlim(0, len(results)); ax.set_ylim(min(results+buyHold), max(results+buyHold))

    bh, = ax.plot([], [], label="Buy & Hold")
    mm, = ax.plot([], [], label="Algorithmic Trading Strategy")
    ax.legend()

    def update(i):
        bh.set_data(range(i), buyHold[:i])
        mm.set_data(range(i), results[:i])
        return bh, mm

    ani = FuncAnimation(fig, update, frames=len(results), interval=30, blit=True)
    
    plt.show()
    return ani  # keep a reference alive

def plotMUD(MUD):
    plt.figure(figsize=(8, 6))
    plt.imshow(MUD, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Transition Probability")

    plt.xticks(ticks=range(8), labels=list(LOOKUP_UD.keys()), rotation=45)
    plt.yticks(ticks=range(8), labels=list(LOOKUP_UD.keys()))

    for i in range(8):
        for j in range(8):
            value = MUD[i, j]
            plt.text(
                j, i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > MUD.max()/2 else "black"
            )

    plt.title("Transition Matrix Heatmap (MUD)")
    plt.xlabel("Back State")
    plt.ylabel("Forward State")
    plt.tight_layout()
    plt.show()

def plotPUD(MUD):
    
    PUD = np.zeros((2,8))
    output = 0

    for i in range(8):
        for j in range (0,4): #up
                PUD[0,i] += MUD[j,i]
        for j in range (4,8): #down
                PUD[1,i] += MUD[j,i]
    
    
    plt.figure(figsize=(8, 6))
    plt.imshow(PUD, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Transition Probability")

    plt.xticks(ticks=range(8), labels=list(LOOKUP_UD.keys()), rotation=45)
    plt.yticks(ticks=range(2), labels=list(["U", "D"]))

    for i in range(2):
        for j in range(8):
            value = PUD[i, j]
            plt.text(
                j, i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > PUD.max()/1.5 else "black"

            )

    plt.title("Probability up or down")
    plt.xlabel("Back State")
    plt.ylabel("Forward State")
    plt.tight_layout()
    plt.show()

def main():
    ticker = "SPY"
    start = "2019-12-30"
    end = "2024-12-30"

    testStart = "2025-01-01"
    testEnd = "2025-12-08"

    getData(ticker,start,end, folder="trainData")
    changes_dict , prices_dict = readData(ticker, folder="trainData")

    getData(ticker, testStart, testEnd, folder="testData")
    changes_dicttest, buyHold = readData(ticker, folder="testData")


    MUD = makeTrMatrix(changes_dict)

    signals = signalGenerator(MUD, changes_dicttest[ticker])
    
    results = backtest(signals, changes_dicttest[ticker])

    buyHoldNorm = []

    for price in buyHold[ticker][:-3]:
        buyHoldNorm.append(price*(10000/(buyHold[ticker][0])))

    plotResultsLive(results, buyHoldNorm, ticker)
    
    plotMUD(MUD)
    plotPUD(MUD)

if __name__ == "__main__":
    main()
