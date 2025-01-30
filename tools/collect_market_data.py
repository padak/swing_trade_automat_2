import os
import signal
import sys
import time
from datetime import datetime
from binance.client import Client
import csv
from pathlib import Path

# --- Configuration ---
API_KEY = os.environ.get('binance_api')
API_SECRET = os.environ.get('binance_secret')
SYMBOL = os.environ.get("SYMBOL", "TRUMPUSDC").strip()
INTERVAL = 10  # seconds between data collection

# Create logs directory if it doesn't exist
Path("./logs").mkdir(parents=True, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"./logs/market_data_{SYMBOL}_{timestamp}.csv"

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handles CTRL+C signal"""
    global running
    print('\nCTRL+C detected. Gracefully shutting down...')
    running = False

def initialize_client():
    """Initialize the Binance client."""
    try:
        client = Client(API_KEY, API_SECRET)
        print("Binance client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return None

def fetch_market_data(client):
    """
    Fetches current market data from Binance.
    Returns a dictionary with the collected data.
    """
    try:
        # Get current price using the same method as in trend_detector_v2_gemini.py
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        
        # Get klines data for additional information
        klines = client.get_klines(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)
        
        # Compile data
        data = {
            'timestamp': datetime.now().isoformat(),
            'price': float(ticker['price']),
            'open': float(klines[0][1]),
            'high': float(klines[0][2]),
            'low': float(klines[0][3]),
            'close': float(klines[0][4]),
            'volume': float(klines[0][5]),
            'close_time': klines[0][6],
            'quote_volume': float(klines[0][7]),
            'trades_count': int(klines[0][8])
        }
        return data
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def setup_csv_file():
    """Sets up the CSV file with headers"""
    try:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                'timestamp', 'price', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume', 'trades_count'
            ]
            writer.writerow(headers)
        print(f"CSV file created: {log_file}")
        return True
    except Exception as e:
        print(f"Error setting up CSV file: {e}")
        return False

def log_data(data):
    """Logs the market data to CSV file"""
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [data[key] for key in data.keys()]
            writer.writerow(row)
    except Exception as e:
        print(f"Error logging data: {e}")

def main():
    print(f"Starting market data collection for {SYMBOL}")
    print(f"Data will be saved to: {log_file}")
    print("Press CTRL+C to stop the collection")

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize Binance client
    client = initialize_client()
    if client is None:
        print("Failed to initialize Binance client. Exiting.")
        return

    # Setup CSV file
    if not setup_csv_file():
        print("Failed to setup CSV file. Exiting.")
        return

    # Main collection loop
    iterations = 0
    while running:
        try:
            data = fetch_market_data(client)
            if data:
                log_data(data)
                iterations += 1
                if iterations % 6 == 0:  # Print status every minute (6 * 10 seconds)
                    print(f"Collected {iterations} data points. Current price: {data['price']}")
            time.sleep(INTERVAL)
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(INTERVAL)

    print(f"\nData collection completed. Collected {iterations} data points.")
    print(f"Data saved to: {log_file}")

if __name__ == "__main__":
    main() 