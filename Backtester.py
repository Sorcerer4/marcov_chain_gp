

def backtest (prices, strategy_function, fee, cash, allow_shorting = False):

    #NOTICE prices is np.array, not array, dict or pd-dataframe

    #Owns no stock from the start
    position = 0
    cash_balance_log = []

    for day in range(len(prices)):
        current_price = prices[day]


        # ----- Determination of trade ------

        target_pos = strategy_function(position, prices, day, cash)

        #Opertunity to cancel unwanted shorting positions
        if not allow_shorting: target_pos = max(target_pos,0)

        trade_amount = target_pos - position

        # Target_pos > pos => trade amount > 0 (BUY)
        # Target_pos < pos => trade amount < 0 (SELL)
        # Target pos < 0 =>  Short position

        # Execute trade
        if trade_amount != 0:
            cost = trade_amount * current_price
            cost += abs(cost) * fee
            cash -= cost
            position = target_pos

        cash_balance_log.append(cash + current_price * position)

    return cash_balance_log
