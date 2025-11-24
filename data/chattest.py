import yfinance as yf
import os
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# DATA FUNCTIONS
# -------------------------
def getData(tickers, start, end, folder="data"):
    os.makedirs(folder, exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)[["Open", "Close"]]
        df["DailyChange"] = (df["Close"] - df["Open"]) / df["Open"]
        df.to_csv(f"{folder}/{ticker}.csv")

def readData(tickers, folder="data"):
    d = {}
    for ticker in tickers:
        df = pd.read_csv(f"{folder}/{ticker}.csv")
        d[ticker] = df["DailyChange"].to_numpy()
    return d

# -------------------------
# MARKOV MATRIX
# -------------------------
def mak_tr_matrix(dailyChanges):
    matrices = {}
    for name, changes in dailyChanges.items():
        M = np.zeros((2, 2))
        count_bull = count_bear = 0

        for i in range(len(changes) - 1):
            if changes[i] < 0 and changes[i+1] < 0:
                M[0,0] += 1
                count_bear += 1
            elif changes[i] < 0 and changes[i+1] > 0:
                M[1,0] += 1
                count_bear += 1
            elif changes[i] > 0 and changes[i+1] < 0:
                M[0,1] += 1
                count_bull += 1
            elif changes[i] > 0 and changes[i+1] > 0:
                M[1,1] += 1
                count_bull += 1

        # normalize
        M[:,0] /= count_bear
        M[:,1] /= count_bull
        matrices[name] = M
    return matrices

# -------------------------
# SIGNAL GENERATION
# -------------------------
def generate_signals(test_changes, matrices, threshold=0.55):
    signals = {}
    for name, changes in test_changes.items():
        M = matrices[name]  # use trained matrix
        sig = []
        for i in range(len(changes)-1):
            today_up = 1 if changes[i] > 0 else 0
            prob_up = M[1, today_up]
            if prob_up > threshold:
                sig.append(1)
            elif prob_up < 1 - threshold:
                sig.append(-1)
            else:
                sig.append(0)
        signals[name] = sig
    return signals

# -------------------------
# BACKTEST
# -------------------------
def backtest(signals, test_changes):
    """
    Compute cumulative returns from signals and test period returns safely.

    Parameters
    ----------
    signals : dict
        Trading signals per stock (-1=sell, 0=hold, 1=buy)
    test_changes : dict
        Daily returns per stock

    Returns
    -------
    results : dict
        Cumulative returns per stock
    """
    results = {}
    for name in signals:
        # Daily returns for test period (shifted to align with signals)
        rets = np.array(test_changes[name][1:])
        rets = np.nan_to_num(rets, nan=0.0)  # replace NaN with 0

        # Corresponding trading signals
        sig = np.array(signals[name])
        sig = np.nan_to_num(sig, nan=0.0)  # replace NaN with 0

        # Element-wise strategy returns
        strat = rets * sig

        # Safe cumulative product
        cum = np.cumprod(1 + strat) - 1
        results[name] = cum

    return results


# -------------------------
# PLOTTING
# -------------------------
def plot_results(matrices, signals, results, tickers):
    for t in tickers:
        M = matrices[t]
        plt.figure(figsize=(4,3))
        sns.heatmap(M, annot=True, fmt=".2f", xticklabels=["Down","Up"], yticklabels=["Down","Up"])
        plt.title(f"Transition matrix — {t}")
        plt.xlabel("Tomorrow")
        plt.ylabel("Today")
        plt.show()

    for t in tickers:
        sig = signals[t]
        plt.figure(figsize=(10,2))
        plt.step(range(len(sig)), sig, where='post')
        plt.ylim([-1.1,1.1])
        plt.title(f"Signals — {t}")
        plt.ylabel("Signal")
        plt.xlabel("Time index")
        plt.show()

    plt.figure(figsize=(10,6))
    for t in tickers:
        plt.plot(results[t], label=t)
    plt.legend()
    plt.title("Cumulative returns (Markov strategy)")
    plt.xlabel("Days")
    plt.ylabel("Cumulative return")
    plt.show()

# -------------------------
# MAIN
#Hej
# -------------------------
def main():
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA","BRK-B","GOOG"]

    # TRAIN DATA
    train_start = "2014-01-01"
    train_end = "2019-12-31"
    getData(tickers, train_start, train_end, folder="train_data")
    train_changes = readData(tickers, folder="train_data")
    matrices = mak_tr_matrix(train_changes)

    # TEST DATA
    test_start = "2020-01-01"
    test_end = date.today().strftime("%Y-%m-%d")
    getData(tickers, test_start, test_end, folder="test_data")
    test_changes = readData(tickers, folder="test_data")

    # generate signals & backtest
    signals = generate_signals(test_changes, matrices, threshold=0.55)
    results = backtest(signals, test_changes)

    # plot
    plot_results(matrices, signals, results, tickers)

    # print final returns
    for t in tickers:
        print(f"{t}: final cumulative return = {results[t][-1]*100:.2f}%")

if __name__ == "__main__":
    main()
