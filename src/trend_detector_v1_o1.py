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

trade_history = []  # We'll store info about each trade in-memory

def record_trade(action, price, threshold_used, profit):
    """
    Log each trade so we can evaluate our strategy afterward.
    """
    trade_history.append({
        "action": action,          # BUY or SELL
        "price": price,           # Price at time of trade
        "threshold": threshold_used,
        "profit": profit,         # Realized profit from that trade, if any
        "timestamp": time.time()  # or datetime.now()
    })

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
                                last_entry_price = current_price
                                last_action = "BUY"

                                # Record the trade, assume no immediate realized profit on a BUY
                                record_trade("BUY", current_price, THRESHOLD_VALUE, 0.0)
                                print(f"[{now_str}] Sim-BUY {bought_amount:.6f} TRUMP at {current_price:.5f}")

                # If trend = DOWN => single SELL
                elif trend == "DOWN" and last_action != "SELL":
                    if trump_balance > 0.000001:
                        gross_usdc = trump_balance * current_price
                        if gross_usdc > 1.2:
                            fee = gross_usdc * 0.001
                            net_usdc = gross_usdc - fee

                            # (Optional) calculate approximate realized profit here.
                            # For a real approach, track cost basis of TRUMP:
                            realized_profit = net_usdc - 500.0  # naive example subtracting the original purchase cost
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
    net_profit = sum(t["profit"] for t in recent_trades)

    # Simple approach: if net_profit is negative, increase threshold. If positive, decrease threshold.
    # This is very naive; you can refine or replace with your own logic!
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

    # If no trades are recorded yet or something else is up, keep the same threshold
    return current_threshold

if __name__ == "__main__":
    main()
