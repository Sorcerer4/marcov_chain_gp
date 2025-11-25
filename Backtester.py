from pygame.examples.go_over_there import target_position


def backtest (prices, strategy_function, fee, cash, allow_shorting = False):

    #Owns no stock from the start
    position = 0
    cash_balance_log = []

    for day in range(len(prices)):
        current_price = prices[day]


        # ----- Determination of trade ------

        target_pos = strategy_function(prices, day, cash)

        #Opertunity to cancel unwanted shorting positions
        if not allow_shorting: target_pos = max(target_pos,0)


        trade_amount = target_pos - position

        # Target_pos > pos => trade amount > 0 (BUY)
        # Target_pos < pos => trade amount < 0 (SELL)
        # Target pos < 0 =>  Short position

        # Execute trade
        if trade_amount != 0:
            cost = trade_amount
