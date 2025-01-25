import os
import time
import math
import argparse
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def main():
    parser = argparse.ArgumentParser(description="Trend Detector v1 - single transaction, with swing threshold.")
    parser.add_argument("--symbol", default="TRUMPUSDC", help="Symbol to trade (default: TRUMPUSDC)")
    args = parser.parse_args()

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

    try:
        while True:
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
                        used_usdc = min(usdc_balance, 200.0)  # example: choose up to 200 USDC
                        if used_usdc < 1.2:
                            print(f"[{now_str}] Skipping BUY - below 1.2 USDC threshold.")
                        else:
                            fee = used_usdc * 0.001
                            spend_usdc = used_usdc - fee
                            if spend_usdc > 0:
                                bought_amount = spend_usdc / current_price
                                trump_balance += bought_amount
                                usdc_balance -= used_usdc
                                last_entry_price = current_price
                                last_action = "BUY"
                                print(f"[{now_str}] Sim-BUY {bought_amount:.6f} TRUMP at {current_price:.5f}, spending {used_usdc:.2f} inc fee={fee:.2f}")

                # If trend = DOWN => single SELL
                elif trend == "DOWN" and last_action != "SELL":
                    if trump_balance > 0.000001:
                        gross_usdc = trump_balance * current_price
                        if gross_usdc < 1.2:
                            print(f"[{now_str}] Skipping SELL - below 1.2 USDC threshold.")
                        else:
                            fee = gross_usdc * 0.001
                            net_usdc = gross_usdc - fee

                            # For realized profit, we need cost basis. We'll skip it or do naive approach:
                            # realizedProfit remains 0 unless we track average cost.
                            usdc_balance += net_usdc
                            trump_balance = 0.0
                            last_entry_price = current_price
                            last_action = "SELL"
                            print(f"[{now_str}] Sim-SELL ALL TRUMP at {current_price:.5f}, gross={gross_usdc:.2f}, fee={fee:.2f}, net={net_usdc:.2f} USDC")

                total_trump_value = trump_balance * current_price
                total_equity = usdc_balance + total_trump_value
                print(
                    f"[{now_str}] usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                    f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                )

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

if __name__ == "__main__":
    main()
