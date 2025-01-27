import os
import time
import math
import argparse
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from scipy import stats
import statistics
import numpy as np
import pmdarima as pm

trade_history = []  # We'll store info about each trade in-memory
total_trump_cost = 500.0  # Track cost basis of TRUMP holdings
realized_profit = 0.0

# Global list to store trade history
trades = []

def record_trade(action, price, threshold_used, profit):
    """
    Log each trade so we can evaluate our strategy afterward.
    """
    trade_dict = {
        "action": action,        # BUY or SELL
        "price": price,          # Price at time of trade
        "threshold": threshold_used,
        "profit": profit if action == "SELL" else 0.0,
        "timestamp": time.time() # or datetime.now()
    }
    trade_history.append(trade_dict)
    print(f"DEBUG: Recorded trade => {trade_dict}")  # debug printing for clarity

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trend Detector v1 - single transaction, with swing threshold.")
    parser.add_argument("--symbol", default="TRUMPUSDC", help="Symbol to trade (default: TRUMPUSDC)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Significance threshold for slope magnitude')
    parser.add_argument(
        "--swing-threshold",
        type=float,
        default=1.0,  # Default threshold; adjust as needed
        help="Threshold for swing detection. Higher means less sensitive to small swings."
    )
    return parser.parse_args()

def main():
    global total_trump_cost  # <-- Now we're using the global variable

    args = parse_arguments()
    global THRESHOLD_VALUE
    THRESHOLD_VALUE = args.swing_threshold

    load_dotenv()  # Load environment if needed for e.g. BINANCE_API_KEY, BINANCE_API_SECRET

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Warning: BINANCE_API_KEY / BINANCE_API_SECRET not found in .env - continuing read-only market data.")
    client = Client(api_key, api_secret)

    symbol = args.symbol

    # ------------------------------------------------------------------------
    # 1) On startup, fetch the *latest* close price so we can set initial TRUMP
    #    We'll pretend we have 500 USDC and 500 USDC worth of TRUMP.
    # ------------------------------------------------------------------------
    start_klines = client.get_klines(symbol=symbol, interval="1m", limit=1)
    if len(start_klines) < 1:
        print("Could not fetch initial price, script will exit.")
        return

    # last close price
    startup_price = float(start_klines[-1][4])
    print(f"Fetched startup price={startup_price:.4f} for {symbol} to initialize balances.")

    # We'll have 500 USDC and the equivalent 500 USDC worth of TRUMP
    usdc_balance = 500.0
    trump_balance = 500.0 / startup_price  # get some TRUMP so we can also test sells
    total_profit_usdc = 0.0  # realized net profit from trades (above the initial holding's cost basis)

    print(f"Initial USDC={usdc_balance:.2f}, TRUMP={trump_balance:.6f} (worth ~500 USDC), total virtual capital=~1000 USDC")

    # Poll interval
    poll_interval_sec = 1

    # Introduce a swing threshold. E.g. 1% difference from our last entry price
    swing_threshold = 0.01  # 1% swing required before next trade
    last_entry_price = startup_price  # track price where we last bought/sold
    last_action = None  # 'BUY' or 'SELL'; helps avoid multiple trades in same direction

    print("Strategy: Single Transaction per Down/Up with a 1% swing threshold.")
    print("Press Ctrl+C to stop.\n")

    iteration_count = 0  # track how many loops have run

    try:
        while True:
            iteration_count += 1
            try:
                klines = client.get_klines(symbol=symbol, interval="1m", limit=25)
                if len(klines) < 25:
                    print("Not enough klines to compute MAs, waiting...")
                    time.sleep(poll_interval_sec)
                    continue

                close_prices = [float(k[4]) for k in klines]
                short_ma = sum(close_prices[-7:]) / 7.0  # MA7
                long_ma  = sum(close_prices) / 25.0      # MA25
                current_price = close_prices[-1]

                # Determine trend
                if short_ma > long_ma:
                    trend = "UP"
                elif short_ma < long_ma:
                    trend = "DOWN"
                else:
                    trend = "FLAT"

                # Time label for logs
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now_str}] MA7={short_ma:.5f} / MA25={long_ma:.5f} => Trend={trend}, Price={current_price:.5f}"
                )

                # Check if price moved enough from our last entry
                price_change_ratio = abs(current_price - last_entry_price) / last_entry_price
                if price_change_ratio < swing_threshold:
                    # Not enough swing to justify a new trade
                    total_trump_value = trump_balance * current_price
                    total_equity = usdc_balance + total_trump_value
                    print(
                        f"[{now_str}] Swing < {swing_threshold*100:.1f}% threshold, no trade. "
                        f"usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                        f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                    )
                    time.sleep(poll_interval_sec)
                    continue

                # If trend = UP => single BUY
                if trend == "UP" and last_action != "BUY":
                    if usdc_balance > 1.2:
                        # e.g., buy up to 200 USDC
                        used_usdc = min(usdc_balance, 200.0)
                        if used_usdc > 1.2:
                            fee = used_usdc * 0.001
                            spend_usdc = used_usdc - fee
                            if spend_usdc > 0:
                                bought_amount = spend_usdc / current_price
                                trump_balance += bought_amount
                                usdc_balance -= used_usdc

                                # Update total_trump_cost with the USDC spent.
                                total_trump_cost += spend_usdc

                                last_entry_price = current_price
                                last_action = "BUY"
                                record_trade("BUY", current_price, THRESHOLD_VALUE, 0.0)
                                print(f"[{now_str}] Sim-BUY {bought_amount:.6f} TRUMP at {current_price:.5f}")

                # If trend = DOWN => single SELL
                elif trend == "DOWN" and last_action != "SELL":
                    if trump_balance > 0.000001:
                        gross_usdc = trump_balance * current_price
                        if gross_usdc > 1.2:
                            fee = gross_usdc * 0.001
                            net_usdc = gross_usdc - fee

                            current_cost_basis = total_trump_cost * (gross_usdc / trump_balance)
                            realized_profit = (current_price * gross_usdc) - current_cost_basis
                            total_trump_cost -= current_cost_basis  # remove sold portion from cost basis

                            usdc_balance += net_usdc
                            trump_balance = 0.0
                            last_entry_price = current_price
                            last_action = "SELL"
                            record_trade("SELL", current_price, THRESHOLD_VALUE, realized_profit)
                            print(f"[{now_str}] Sim-SELL ALL TRUMP at {current_price:.5f}, netUSDC={net_usdc:.2f} (profit={realized_profit:.2f})")

                total_trump_value = trump_balance * current_price
                total_equity = usdc_balance + total_trump_value
                print(
                    f"[{now_str}] usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                    f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                )

                # Periodically re-calibrate threshold (e.g., every 10 loops)
                if iteration_count % 10 == 0:
                    THRESHOLD_VALUE = auto_calibrate_threshold(THRESHOLD_VALUE)

                time.sleep(poll_interval_sec)

            except (BinanceAPIException, BinanceRequestException) as e:
                print(f"Error fetching market data or parsing results: {e}")
                time.sleep(5)

    except KeyboardInterrupt:
        final_trump_value = trump_balance * current_price
        final_total = usdc_balance + final_trump_value
        print("\nShutting down read-only simulation...")
        print(f"Final USDC={usdc_balance:.2f}, TRUMP={trump_balance:.6f} (~{final_trump_value:.2f} USDC), total equity={final_total:.2f} USDC")
        print(f"Realized net profit recorded={total_profit_usdc:.2f} USDC")

    # Then you can log or print the total realized PnL:
    current_realized_pnl = calculate_realized_pnl()
    print(f"Current realized PnL: {current_realized_pnl:.2f} USDC")

def detect_trends(input_file):
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # Add explicit format
        # ... existing code ...
        result = {
            'trend_direction': trend_direction,
            'slope': slope,
            'p_value': p_value,
            'r_squared': model.score(X, y),  # New metric
            'confidence_interval': {
                'lower': conf_int[0][0],
                'upper': conf_int[0][1]
            }
        }
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Empty CSV file")
        return None

def check_normality(residuals):
    """Check normality using Shapiro-Wilk test"""
    stat, p = stats.shapiro(residuals)
    return p > 0.05

def detect_swing(current_price, recent_price):
    """
    Example function: returns True if the price has moved more
    than 'threshold_value' percent from recent price, else False.
    """
    # You can implement your own logic here. This is just an example.
    if recent_price == 0:
        return False
    move_percent = abs((current_price - recent_price) / recent_price) * 100
    return move_percent > THRESHOLD_VALUE

def trade_on_swing(binance_client, current_price, current_holdings_usdc, current_holdings_trump, threshold_value):
    """
    Decides what trade to make based on the current trend or swing detection.
    Returns updated USDC and TRUMP balances.
    """
    # Example logic: compare 'current_price' to a short-term average to detect a swing
    downward_swing = (current_price < short_term_avg_price - threshold_value)
    upward_swing   = (current_price > short_term_avg_price + threshold_value)

    # Keep base holdings locked in
    base_usdc  = 500  # minimum USDC
    base_trump = 500 / current_price  # 500 USDC worth of TRUMP

    if downward_swing:
        # Place one large sell order for any TRUMP above the base holding
        trump_to_sell = current_holdings_trump - base_trump
        if trump_to_sell > 0:
            # Sell in a single transaction
            order = binance_client.order_market_sell(symbol='TRUMPUSDC', quantity=trump_to_sell)
            # Re-check updated balances from the order result if desired...

    elif upward_swing:
        # Example: use available USDC above base to buy more TRUMP
        usdc_available = current_holdings_usdc - base_usdc
        if usdc_available > 0:
            # Buy TRUMP in a single transaction
            trump_to_buy = usdc_available / current_price  # approximate
            order = binance_client.order_market_buy(symbol='TRUMPUSDC', quantity=trump_to_buy)
            # Re-check updated balances if needed...

    # ... return updated balances ...
    return current_holdings_usdc, current_holdings_trump

def auto_calibrate_threshold(current_threshold):
    """
    Example logic: reviews the last 5 trades recorded, sees if net profit
    is positive or negative, and adjusts threshold up/down by a small step.
    """
    recent_trades = trade_history[-5:]  # last 5 trades
    # Only consider profit from SELL trades
    net_profit = sum(t["profit"] for t in recent_trades if t["action"] == "SELL")

    if len(recent_trades) > 0:
        if net_profit < 0:
            new_threshold = current_threshold + 0.1
            print(f"auto_calibrate_threshold: net_profit={net_profit:.2f}, "
                  f"increasing threshold from {current_threshold:.2f} to {new_threshold:.2f}")
        else:
            new_threshold = max(0.1, current_threshold - 0.1)  # don't go below 0.1
            print(f"auto_calibrate_threshold: net_profit={net_profit:.2f}, "
                  f"lowering threshold from {current_threshold:.2f} to {new_threshold:.2f}")
        return new_threshold

    return current_threshold  # If no trades or something else, keep existing threshold

def train_auto_arima_model(historical_prices):
    """
    Trains an AutoARIMA model on the provided historical price data.
    Returns the fitted model that can be used to forecast future prices.
    """
    # Convert prices to a pandas Series if not already
    price_series = pd.Series(historical_prices)

    # Fit AutoARIMA model; it will auto-adjust parameters like p, d, q, etc.
    # You may wish to adjust parameters such as seasonal=False, stepwise=True, etc. to suit your data
    model = pm.auto_arima(
        price_series,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True
    )
    return model

def decide_trade(current_price, model, steps_ahead=1):
    """
    Predicts future price based on the fitted model and decides whether to 
    buy, sell, or hold. For simplicity, we look 1 step ahead.
    """
    forecast = model.predict(n_periods=steps_ahead)
    predicted_price = forecast[-1]

    # Basic logic demonstration: 
    # If predicted price is above current price, consider BUY (or hold long).
    # If predicted price is below current price, consider SELL (or reduce position).
    if predicted_price > current_price:
        return "BUY"
    else:
        return "SELL"

def run_trading_strategy():
    """
    Main method (example) for running the auto-adjusting strategy. 
    It uses historical data to train the model, then checks current price 
    to decide on trades.
    """
    # Step 1: Retrieve historical data (closing prices)
    # This might already exist in your script; we demonstrate a placeholder here
    historical_prices = get_historical_price_data("TRUMP", limit=200)

    # Step 2: Train the auto-arima model
    arima_model = train_auto_arima_model(historical_prices)

    # Step 3: Get current data
    current_price = get_current_price("TRUMP")

    # Step 4: Decide trade
    trade_action = decide_trade(current_price, arima_model, steps_ahead=1)

    # Step 5: Place order based on the decision
    if trade_action == "BUY":
        # Place a BUY order logic
        # e.g., place_buy_order("TRUMP", quantity, current_price)
        pass
    else:
        # Place a SELL order logic
        # e.g., place_sell_order("TRUMP", quantity, current_price)
        pass

    # ... remaining code for handling orders, logging, etc.

def execute_trade(signal, current_price, usdc_balance, trump_balance):
    """
    Helper function to place trades based on the detected signal.
    For demonstration, using a simple immediate execute approach.
    """
    order_qty = 0
    trade_type = "NEUTRAL"
    order_status = "NONE"

    # Example logic:
    if signal == "BUY":
        trade_type = "BUY"
        # For example, use half of the USDC balance to buy TRUMP
        trade_amount_usdc = usdc_balance / 2
        if trade_amount_usdc >= 1.0: # ensure min trade size
            order_qty = trade_amount_usdc / current_price
            usdc_balance -= trade_amount_usdc
            trump_balance += order_qty
            order_status = "FILLED"

    elif signal == "SELL":
        trade_type = "SELL"
        # For example, sell half of TRUMP holdings
        sell_qty = trump_balance / 2
        if sell_qty * current_price >= 1.0: # ensure min trade size
            order_qty = sell_qty
            trump_balance -= sell_qty
            usdc_balance += sell_qty * current_price
            order_status = "FILLED"

    # Store trade details in the trades list
    if order_status == "FILLED":
        trades.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "signal": signal,
            "trade_type": trade_type,
            "qty": order_qty,
            "price": current_price,
            "usdc_balance_after": usdc_balance,
            "trump_balance_after": trump_balance,
            "status": order_status
        })

    return usdc_balance, trump_balance

# Optional: function to calculate realized PnL
def calculate_realized_pnl():
    """
    Example PnL calculation that pairs each SELL with its preceding BUY.
    This is very simplistic and may need refinement for partial fills or multiple open orders.
    """
    realized_pnl = 0.0
    last_buy = None

    for trade in trades:
        if trade["trade_type"] == "BUY":
            last_buy = trade
        elif trade["trade_type"] == "SELL" and last_buy:
            # Realized PnL = (sell_price - buy_price) * qty
            realized_pnl += (trade["price"] - last_buy["price"]) * trade["qty"]
            last_buy = None
    return realized_pnl

if __name__ == "__main__":
    main()
