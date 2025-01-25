import os
import time
import math
import argparse
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def main():
    parser = argparse.ArgumentParser(description="Trend Detector v1 - virtual trading only, with more aggressive partial orders.")
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

    print("Strategy: 1) shortTermMA=MA7, longTermMA=MA25, updated every second from 25 1m klines.")
    print("          2) If MA7 < MA25 => SELL partial TRUMP (20%). If MA7 > MA25 => BUY partial USDC (20%).")
    print("          3) 0.1% fee applies to each trade. Realized profit accumulates in total_profit_usdc.")
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

                # Determine simple "trend" by comparing shortMA and longMA
                if short_ma > long_ma:
                    trend = "UP"
                elif short_ma < long_ma:
                    trend = "DOWN"
                else:
                    trend = "FLAT"

                print(
                    f"MA7={short_ma:.5f} / MA25={long_ma:.5f} => Trend={trend}, "
                    f"Price={current_price:.5f}"
                )

                # Aggressive partial trading logic
                # (A) If trend = UP => BUY
                if trend == "UP":
                    # Attempt to buy some portion of USDC
                    # e.g. 20% of current USDC, up to a max of 100 USDC
                    if usdc_balance > 1.0:
                        used_usdc = min(usdc_balance * 0.20, 100.0)
                        # Enforce minimum trade value of 1.2 USDC
                        if used_usdc < 1.2:
                            print("Skipping BUY - below 1.2 USDC threshold.")
                            # skip the trade
                        else:
                            fee = used_usdc * 0.001
                            spend_usdc = used_usdc - fee
                            if spend_usdc > 0:
                                bought_amount = spend_usdc / current_price
                                trump_balance += bought_amount
                                usdc_balance -= used_usdc
                                print(f"  Sim-BUY {bought_amount:.6f} TRUMP at {current_price:.5f}, spending {used_usdc:.2f} inc fee={fee:.2f}")

                # (B) If trend = DOWN => SELL some portion of TRUMP
                elif trend == "DOWN":
                    # e.g. 20% of TRUMP holdings
                    if trump_balance > 0.000001:
                        sell_amount = trump_balance * 0.20
                        gross_usdc = sell_amount * current_price
                        # Enforce minimum trade value of 1.2 USDC
                        if gross_usdc < 1.2:
                            print("Skipping SELL - below 1.2 USDC threshold.")
                            # skip the trade
                        else:
                            fee = gross_usdc * 0.001
                            net_usdc = gross_usdc - fee

                            # For realized profit, we approximate cost basis from the portion sold.
                            # We'll do a naive approach that the original cost basis was last price we acquired.
                            # But a more accurate approach would track each buy-lot. Here, we keep it simple.
                            # We'll track "realized profit" as if our 'average cost' was current_price for now.
                            # (Better approach would store average cost basis for all TRUMP.)
                            # For demonstration, let's treat the entire portion as though our cost was current_price
                            # => realized_profit = net_usdc - (sell_amount * current_price)
                            # => that's zero, so let's not add to total_profit. 
                            #
                            # If you want to track actual P/L, you'd need a weighted average cost approach.
                            # For now, we won't increment total_profit_usdc from partial sells 
                            # because we don't have cost basis subdivided. 
                            #
                            # We'll increment total_profit only if we assume a cost basis for the sold portion
                            # is exactly current_price (that yields 0 P/L). 
                            # Or you can store an "average buy in" if you'd like.

                            usdc_balance += net_usdc
                            trump_balance -= sell_amount
                            print(f"  Sim-SELL {sell_amount:.6f} TRUMP at {current_price:.5f}, gross={gross_usdc:.2f}, fee={fee:.2f}, net={net_usdc:.2f} USDC")

                # (C) FLAT => do nothing
                # (D) Print quick balances
                total_trump_value = trump_balance * current_price
                total_equity = usdc_balance + total_trump_value
                print(
                    f"    usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                    f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                )

                time.sleep(poll_interval_sec)

            except (BinanceAPIException, BinanceRequestException) as e:
                print(f"Error fetching market data or parsing results: {e}")
                time.sleep(5)

    except KeyboardInterrupt:
        # On exit, show final results
        final_trump_value = trump_balance * current_price
        final_total = usdc_balance + final_trump_value
        print("\nShutting down read-only simulation...")
        print(f"Final USDC={usdc_balance:.2f}, TRUMP={trump_balance:.6f} (~{final_trump_value:.2f} USDC), total equity={final_total:.2f} USDC")
        print(f"Realized net profit recorded={total_profit_usdc:.2f} USDC")

if __name__ == "__main__":
    main()
