import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons


def plot_interactive_with_normalize(results_dict, default_normalize=True):
    """
    Interaktiv plott med RadioButtons (ticker) och CheckButtons (normalize toggle)

    :param results_dict: dict med ticker -> equity_curve (list eller numpy array)
    :param default_normalize: bool, om normalize är påslagen vid start
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.35)  # mer plats åt widgets

    tickers = list(results_dict.keys())
    current_ticker = tickers[0]
    normalize = default_normalize

    # --- initial plott ---
    curve = results_dict[current_ticker]
    if normalize:
        curve = curve / curve[0]
    line, = ax.plot(curve, lw=2)

    ax.set_title(current_ticker)
    ax.set_xlabel("Days")
    ax.set_ylabel("Normalized Portfolio Value" if normalize else "Portfolio Value")
    ax.grid(True, alpha=0.3)

    # --- RadioButtons för tickers ---
    axcolor = 'lightgoldenrodyellow'
    rax = plt.axes([0.02, 0.4, 0.25, 0.5], facecolor=axcolor)  # flyttad åt vänster
    radio = RadioButtons(rax, tickers)

    # --- CheckButtons för normalize ---
    cax = plt.axes([0.02, 0.3, 0.25, 0.1], facecolor=axcolor)
    check = CheckButtons(cax, ['Normalize'], [normalize])

    # --- funktioner för uppdatering ---
    def update_plot():
        curve = results_dict[current_ticker]
        if normalize:
            curve = curve / curve[0]
        line.set_ydata(curve)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(current_ticker)
        ax.set_ylabel("Normalized Portfolio Value" if normalize else "Portfolio Value")
        fig.canvas.draw_idle()

    def update_ticker(label):
        nonlocal current_ticker
        current_ticker = label
        update_plot()

    def update_normalize(label):
        nonlocal normalize
        normalize = not normalize
        update_plot()

    radio.on_clicked(update_ticker)
    check.on_clicked(update_normalize)

    plt.show()


def plot_all(results_dict, normalize=True):
    """
    Plottar alla tickers samtidigt på en graf.

    :param results_dict: dict med ticker -> equity_curve
    :param normalize: bool, om True, normaliseras kurvorna till startvärde 1
    """
    plt.figure(figsize=(12, 6))

    for ticker, curve in results_dict.items():
        curve_to_plot = curve
        if normalize:
            curve_to_plot = curve / curve[0]
        plt.plot(curve_to_plot, label=ticker, lw=1.8)

    plt.title("Alla aktier – jämförande equity curves")
    plt.xlabel("Days")
    plt.ylabel("Normalized Portfolio Value" if normalize else "Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()