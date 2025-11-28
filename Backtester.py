def backtest (ticker, prices,matrix, strategy_fn, fee, cash, allow_shorting):

    #NOTICE prices is np.array, not array, dict or pd-dataframe

    #Owns no stock from the start
    position = 0
    cash_balance_log = []

    for day in range(len(prices)):
        current_price = prices[day]

        # ----- Determination of trade ------

        target_pos = strategy_fn(ticker, position, prices, day, cash, matrix)

        #Opertunity to cancel unwanted shorting positions
        if not allow_shorting: target_pos = max(target_pos,0)

        trade_amount = target_pos - position

        # Target_pos > pos => trade amount > 0 (BUY)
        # Target_pos < pos => trade amount < 0 (SELL)
        # Target pos < 0 =>  Short position

        # Execute trade
        if trade_amount != 0:

            # Count preliminary cost (before capacity controll)
            cost = trade_amount * current_price
            cost += abs(cost) * fee

            # We want to buy but don't have money
            #if trade_amount > 0 and cost > cash:
             #   max_affordable = int(cash / (current_price * (1 + fee)))
              #  trade_amount = max_affordable

                # Update cost after change
               # cost = trade_amount * current_price
                #cost += abs(cost) * fee
            #hej

            # Do the trade
            cash -= cost
            position += trade_amount

        cash_balance_log.append(cash + current_price * position)
    return cash_balance_log