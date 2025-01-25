import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

def main():
    # Load environment variables from .env file
    load_dotenv()

    # 1. Retrieve API credentials from environment variables
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY and/or BINANCE_API_SECRET not set in environment.")
        return

    # 2. Initialize the Binance Client
    client = Client(api_key, api_secret)

    try:
        # 3. Fetch account information
        account_info = client.get_account()
        balances = account_info.get("balances", [])
        print("Account Balances (non-zero):")
        for balance in balances:
            if float(balance["free"]) > 0:
                print(f"  Asset: {balance['asset']} | Free: {balance['free']} | Locked: {balance['locked']}")

        # 4. Fetch open orders for TRUMPUSDC (or your chosen symbol)
        symbol = "TRUMPUSDC"
        open_orders = client.get_open_orders(symbol=symbol)
        if open_orders:
            print(f"\nOpen Orders for {symbol}:")
            for order in open_orders:
                print(f"  Order ID: {order['orderId']} | Side: {order['side']} | Price: {order['price']} "
                      f"| OrigQty: {order['origQty']} | Status: {order['status']}")
        else:
            print(f"\nNo open orders for {symbol}.")

    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"Binance API error: {e}")

if __name__ == "__main__":
    main() 