import os
import time
from dotenv import load_dotenv
import argparse
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def main():
    parser = argparse.ArgumentParser(description="Version 3: integrates BUY→SELL & SELL→BUY logic with MA7.")
    parser.add_argument("--dry-run", action="store_true", help="If set, do not place any orders on Binance, only print them.")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY and/or BINANCE_API_SECRET not set.")
        return

    client = Client(api_key, api_secret)

    symbol = "TRUMPUSDC"
    poll_interval_seconds = 1  # how often to poll for new trades

    # ---------------------------------------------------------------------------
    # 1. Gather existing trades at startup so we don't place orders for old trades
    # ---------------------------------------------------------------------------
    old_buy_ids = set()
    old_sell_ids = set()
    try:
        existing_trades = client.get_my_trades(symbol=symbol)
        for t in existing_trades:
            # t["isBuyer"] == True means a BUY trade
            # t["isBuyer"] == False means a SELL trade
            if t["isBuyer"]:
                old_buy_ids.add(t["id"])
            else:
                old_sell_ids.add(t["id"])
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"Error fetching initial trades: {e}")

    # We'll track any trades for which we've already triggered an opposite order
    triggered_buys = set()   # For new BUY trades that triggered SELL
    triggered_sells = set()  # For new SELL trades that triggered BUY

    # Keep track of previous MA7 so we know if it's going UP or DOWN
    prev_ma7 = None

    # We'll store up to 15 lines of MA7 logs here
    ma7_history = []

    def get_direction_label(curr: float, prev: float) -> str:
        """Return 'UP', 'DOWN', or 'FLAT' if we have a previous value."""
        if prev is None:
            return "N/A"
        if curr > prev:
            return "UP"
        elif curr < prev:
            return "DOWN"
        else:
            return "FLAT"

    if args.dry_run:
        print(f"Starting V3 main loop for {symbol} in DRY-RUN mode. Polling every {poll_interval_seconds}s...")
    else:
        print(f"Starting V3 main loop for {symbol}. Polling every {poll_interval_seconds}s...")

    # We'll handle Ctrl+C gracefully
    try:
        # Next time to compute MA7 (5 seconds from now)
        next_ma7_update = time.time() + 5

        while True:
            try:
                # -------------------------------------------------
                # (A) Compute & print MA7 every 5 seconds
                # -------------------------------------------------
                curr_time = time.time()
                if curr_time >= next_ma7_update:
                    klines = client.get_klines(symbol=symbol, interval="1m", limit=7)
                    if len(klines) < 7:
                        print("Warning: not enough klines to compute MA7.")
                    else:
                        # Compute MA7
                        close_prices = [float(k[4]) for k in klines]  # close = k[4]
                        current_ma7 = sum(close_prices) / 7.0

                        # Determine direction
                        direction = get_direction_label(current_ma7, prev_ma7)
                        prev_ma7 = current_ma7

                        # Format a single-line log for MA7
                        ma7_log = f"MA7={current_ma7:.6f}({direction})"

                        # Add to ring buffer; keep only last 15
                        ma7_history.append(ma7_log)
                        if len(ma7_history) > 15:
                            ma7_history.pop(0)

                        # Print the entire ring buffer (MA7 logs)
                        for line in ma7_history:
                            print(line)
                        # Print a divider
                        print("-----")

                    # Schedule the next MA7 update for 5 seconds later
                    next_ma7_update += 5

                # -------------------------------------------------
                # (B) Poll trades (both BUY & SELL)
                # -------------------------------------------------
                trades = client.get_my_trades(symbol=symbol)
                for trade in trades:
                    trade_id = trade["id"]
                    is_buyer = trade["isBuyer"]  # True=BUY, False=SELL
                    fill_price = float(trade["price"])
                    qty = float(trade["qty"])

                    # ---------------------------------------------
                    # (B1): If BUY, place a SELL at +1%
                    # ---------------------------------------------
                    if is_buyer:
                        # Skip if old or if SELL was already triggered
                        if trade_id in old_buy_ids or trade_id in triggered_buys:
                            continue

                        print(f"Detected NEW BUY fill: Trade ID={trade_id}, Price={fill_price}, Qty={qty}")
                        sell_price = round(fill_price * 1.01, 6)
                        print(f"Calculated SELL at {sell_price} for quantity={qty} (+1%).")

                        if args.dry_run:
                            print(f"[DRY-RUN] Would have placed LIMIT SELL at {sell_price}.")
                        else:
                            try:
                                order = client.create_order(
                                    symbol=symbol,
                                    side="SELL",
                                    type="LIMIT",
                                    timeInForce="GTC",
                                    quantity=qty,
                                    price=str(sell_price)
                                )
                                print(f"SELL order created: Order ID={order['orderId']}")
                            except (BinanceAPIException, BinanceRequestException) as e:
                                print(f"Failed to place SELL order (BUY fill) {trade_id}: {e}")

                        triggered_buys.add(trade_id)

                    # ---------------------------------------------
                    # (B2): If SELL, place a BUY at -1%
                    # ---------------------------------------------
                    else:  # is_buyer == False => SELL trade
                        # Skip if old or if BUY was already triggered
                        if trade_id in old_sell_ids or trade_id in triggered_sells:
                            continue

                        print(f"Detected NEW SELL fill: Trade ID={trade_id}, Price={fill_price}, Qty={qty}")
                        buy_price = round(fill_price * 0.99, 6)
                        print(f"Calculated BUY at {buy_price} for quantity={qty} (-1%).")

                        if args.dry_run:
                            print(f"[DRY-RUN] Would have placed LIMIT BUY at {buy_price}.")
                        else:
                            try:
                                order = client.create_order(
                                    symbol=symbol,
                                    side="BUY",
                                    type="LIMIT",
                                    timeInForce="GTC",
                                    quantity=qty,
                                    price=str(buy_price)
                                )
                                print(f"BUY order created: Order ID={order['orderId']}")
                            except (BinanceAPIException, BinanceRequestException) as e:
                                print(f"Failed to place BUY order (SELL fill) {trade_id}: {e}")

                        triggered_sells.add(trade_id)

                # -------------------------------------------------
                # (C) Sleep until next iteration
                # -------------------------------------------------
                time.sleep(poll_interval_seconds)

            except (BinanceAPIException, BinanceRequestException) as e:
                print(f"Error fetching trades or placing order: {e}")
                time.sleep(30)  # wait longer if error

    except KeyboardInterrupt:
        print("\nShutting down gracefully... Bye!")

if __name__ == "__main__":
    main()