import os
import time
from binance.client import Client
from binance.enums import *
import numpy as np

# Documentation: https://python-binance.readthedocs.io/en/latest/

# --- Configuration ---
API_KEY = os.environ.get('binance_api')
API_SECRET = os.environ.get('binance_secret')
SYMBOL = 'TRUMPUSDC'
INITIAL_USDC = 500
INITIAL_TRUMP_USDC_VALUE = 500
MIN_TRADE_USDC = 1.2

# --- Parameters for Self-Learning (Initial values, will be adjusted) ---
FAST_MA_PERIOD = 12  # Initial period for fast moving average
SLOW_MA_PERIOD = 26  # Initial period for slow moving average
RSI_PERIOD = 14       # Initial period for RSI
RSI_OVERBOUGHT = 70  # Initial RSI overbought level
RSI_OVERSOLD = 30   # Initial RSI oversold level
STOP_LOSS_PERCENT = 0.05 # Initial stop loss percentage (5%)
TAKE_PROFIT_PERCENT = 0.10 # Initial take profit percentage (10%)

# --- Global Variables ---
client = None
initial_trump_qty = 0.0
current_strategy_params = {}
historical_performance = []


def initialize_client():
    """Initialize the Binance client."""
    global client
    client = Client(API_KEY, API_SECRET)
    print("Binance client initialized.")

def fetch_current_price(symbol):
    """Fetch current price of a symbol."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def calculate_initial_trump_quantity(initial_usdc_value):
    """Calculate initial TRUMP quantity based on current price."""
    current_price = fetch_current_price(SYMBOL)
    if current_price:
        return initial_usdc_value / current_price
    return 0.0

def fetch_historical_data(symbol, interval, period):
    """Fetch historical klines data for a symbol."""
    try:
        klines = client.get_historical_klines(symbol, interval, period)
        # Extract closing prices
        closing_prices = np.array([float(kline[4]) for kline in klines])
        return closing_prices
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

def calculate_moving_averages(prices, fast_period, slow_period):
    """Calculate fast and slow moving averages."""
    fast_ma = np.mean(prices[-fast_period:]) if len(prices) >= fast_period else np.array([])
    slow_ma = np.mean(prices[-slow_period:]) if len(prices) >= slow_period else np.array([])
    return fast_ma, slow_ma

def calculate_rsi(prices, period):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return np.array([])

    price_diffs = np.diff(prices)
    gains = price_diffs[price_diffs > 0]
    losses = np.abs(price_diffs[price_diffs < 0])

    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.array([0])
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.array([0.000000001]) # avoid division by zero

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_trading_signal(prices, fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold):
    """Generate trading signal based on moving averages and RSI."""
    fast_ma, slow_ma = calculate_moving_averages(prices, fast_ma_period, slow_ma_period)
    rsi_value = calculate_rsi(prices, rsi_period)

    if not fast_ma.size or not slow_ma.size or not rsi_value.size:
        return "NEUTRAL"

    if fast_ma > slow_ma and rsi_value < rsi_oversold:
        return "BUY"
    elif fast_ma < slow_ma and rsi_value > rsi_overbought:
        return "SELL"
    else:
        return "NEUTRAL"

def execute_trade(signal, current_price, usdc_balance, trump_balance):
    """Simulate trade execution."""
    trade_qty = 0.0

    if signal == "BUY":
        trade_qty_usdc = usdc_balance * 0.99 # Use 99% of available USDC
        if trade_qty_usdc > MIN_TRADE_USDC:
            trade_qty = trade_qty_usdc / current_price
            print(f"BUY {SYMBOL}: Quantity: {trade_qty:.2f}, Price: {current_price}")
            return "BUY", trade_qty
        else:
            print(f"BUY signal, but trade quantity below minimum. USDC balance: {usdc_balance}")
            return "NEUTRAL", 0.0

    elif signal == "SELL":
        trade_qty = trump_balance
        if trade_qty * current_price > MIN_TRADE_USDC:
            print(f"SELL {SYMBOL}: Quantity: {trade_qty:.2f}, Price: {current_price}")
            return "SELL", trade_qty
        else:
            print(f"SELL signal, but trade value below minimum. TRUMP balance value: {trump_balance * current_price}")
            return "NEUTRAL", 0.0
    return "NEUTRAL", 0.0


def update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance):
    """Update USDC and TRUMP balances based on trade execution."""
    if trade_action == "BUY":
        usdc_balance -= trade_qty * current_price
        trump_balance += trade_qty
    elif trade_action == "SELL":
        usdc_balance += trade_qty * current_price
        trump_balance -= trade_qty
    return usdc_balance, trump_balance

def evaluate_performance(initial_portfolio_value, current_portfolio_value):
    """Evaluate performance and calculate profit/loss percentage."""
    profit_loss = current_portfolio_value - initial_portfolio_value
    profit_loss_percent = (profit_loss / initial_portfolio_value) * 100
    return profit_loss_percent

def adjust_strategy_parameters(performance_percent):
    """Adjust strategy parameters based on performance."""
    global FAST_MA_PERIOD, SLOW_MA_PERIOD, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT

    if performance_percent > 0.5: # If performance is good, try to explore more
        FAST_MA_PERIOD = max(5, FAST_MA_PERIOD - 1) # Faster MA
        RSI_OVERBOUGHT = min(80, RSI_OVERBOUGHT + 1) # Higher overbought
        RSI_OVERSOLD = max(20, RSI_OVERSOLD - 1)     # Lower oversold
        print("Improved performance, adjusting parameters to explore further.")
    elif performance_percent < -1.0: # If performance is bad, adjust more aggressively
        SLOW_MA_PERIOD = min(50, SLOW_MA_PERIOD + 2) # Slower MA
        RSI_OVERBOUGHT = max(60, RSI_OVERBOUGHT - 2) # Lower overbought
        RSI_OVERSOLD = min(40, RSI_OVERSOLD + 2)     # Higher oversold
        STOP_LOSS_PERCENT = min(0.10, STOP_LOSS_PERCENT + 0.01) # Wider stop loss
        print("Poor performance, adjusting parameters to be more conservative.")
    else:
        print("Moderate performance, minor parameter adjustments.")
        FAST_MA_PERIOD = max(5, FAST_MA_PERIOD -0.5) # Slightly faster MA
        SLOW_MA_PERIOD = min(50, SLOW_MA_PERIOD + 0.5) # Slightly slower MA


    current_strategy_params = {
        "FAST_MA_PERIOD": FAST_MA_PERIOD,
        "SLOW_MA_PERIOD": SLOW_MA_PERIOD,
        "RSI_PERIOD": RSI_PERIOD,
        "RSI_OVERBOUGHT": RSI_OVERBOUGHT,
        "RSI_OVERSOLD": RSI_OVERSOLD,
        "STOP_LOSS_PERCENT": STOP_LOSS_PERCENT,
        "TAKE_PROFIT_PERCENT": TAKE_PROFIT_PERCENT
    }
    print(f"New strategy parameters: {current_strategy_params}")


def main():
    global initial_trump_qty, current_strategy_params, historical_performance

    initialize_client()
    initial_trump_qty = calculate_initial_trump_quantity(INITIAL_TRUMP_USDC_VALUE)

    if initial_trump_qty == 0.0:
        print("Could not fetch initial TRUMP price. Exiting.")
        return

    usdc_balance = INITIAL_USDC
    trump_balance = initial_trump_qty
    prices_history = []
    initial_portfolio_value = INITIAL_USDC + INITIAL_TRUMP_USDC_VALUE # Initial portfolio value in USDC

    current_strategy_params = { # Store initial parameters
        "FAST_MA_PERIOD": FAST_MA_PERIOD,
        "SLOW_MA_PERIOD": SLOW_MA_PERIOD,
        "RSI_PERIOD": RSI_PERIOD,
        "RSI_OVERBOUGHT": RSI_OVERBOUGHT,
        "RSI_OVERSOLD": RSI_OVERSOLD,
        "STOP_LOSS_PERCENT": STOP_LOSS_PERCENT,
        "TAKE_PROFIT_PERCENT": TAKE_PROFIT_PERCENT
    }


    print(f"Starting with: {usdc_balance:.2f} USDC, {trump_balance:.2f} TRUMP")
    print(f"Initial strategy parameters: {current_strategy_params}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        current_price = fetch_current_price(SYMBOL)
        if not current_price:
            time.sleep(60) # Wait for 1 minute before retrying
            continue

        prices_history.append(current_price) # Append current price to history

        if len(prices_history) > SLOW_MA_PERIOD: # Only trade after enough data is collected
            signal = generate_trading_signal(
                prices_history,
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"],
                current_strategy_params["RSI_OVERBOUGHT"],
                current_strategy_params["RSI_OVERSOLD"]
            )
            print(f"Current Price: {current_price}, Signal: {signal}")

            trade_action, trade_qty = execute_trade(signal, current_price, usdc_balance, trump_balance)
            if trade_action != "NEUTRAL":
                usdc_balance, trump_balance = update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance)


        current_portfolio_value = usdc_balance + (trump_balance * current_price)
        performance_percent = evaluate_performance(initial_portfolio_value, current_portfolio_value)

        print(f"USDC Balance: {usdc_balance:.2f}, TRUMP Balance: {trump_balance:.2f}, Portfolio Value: {current_portfolio_value:.2f} USDC, P/L: {performance_percent:.2f}%")

        historical_performance.append(performance_percent) # Store performance history

        if iteration % 10 == 0: # Adjust parameters every 10 iterations
            if len(historical_performance) >= 10: # Adjust based on last 10 iterations average performance
                avg_performance = np.mean(historical_performance[-10:])
                print(f"Average performance over last 10 iterations: {avg_performance:.2f}%")
                adjust_strategy_parameters(avg_performance)


        time.sleep(10) # Check price every 10 seconds


if __name__ == "__main__":
    main() 