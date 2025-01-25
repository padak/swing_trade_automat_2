import signal # For signal handling
import os
import time
from binance.client import Client
from binance.enums import *
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.linear_model import LogisticRegression # Example ML model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Signal module imported successfully:", 'signal' in globals()) # Check if signal is imported

# Documentation: https://python-binance.readthedocs.io/en/latest/

# --- Configuration ---
API_KEY = os.environ.get('binance_api')
API_SECRET = os.environ.get('binance_secret')
SYMBOL_raw = os.environ.get("SYMBOL", "TRUMPUSDC")  # Default to "TRUMPUSDC" if not set (still with space for now, corrected below)
SYMBOL = SYMBOL_raw.strip() # Remove leading/trailing whitespace, important for API calls
INITIAL_USDC = 500
INITIAL_TRUMP_USDC_VALUE = 500
MIN_TRADE_USDC = 1.2

# --- Parameters for Self-Learning (Initial values, will be adjusted) ---
FAST_MA_PERIOD_DEFAULT = 5  # default value if env var is not set or invalid (Shorter period)
SLOW_MA_PERIOD_DEFAULT = 20 # default value if env var is not set or invalid (Shorter period)
RSI_PERIOD = 14       # Initial period for RSI
RSI_OVERBOUGHT = 70  # Initial RSI overbought level
RSI_OVERSOLD = 30   # Initial RSI oversold level
STOP_LOSS_PERCENT = float(os.environ.get("STOP_LOSS_PERCENT", 0.001)) # e.g., 0.001 for 0.1%
TAKE_PROFIT_PERCENT = 0.10 # Initial take profit percentage (10%)

# --- Global Variables ---
client = None
initial_trump_qty = 0.0
current_strategy_params = {}
historical_performance = []
trading_enabled = True # Flag for trading loop - initialize here

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_SECRET = os.environ.get("GEMINI_API_SECRET")
INVEST_USD = float(os.environ.get("INVEST_USD", 500))  # Default to 500 USD if not set
PROFIT_THRESHOLD_PERCENT = float(os.environ.get("PROFIT_THRESHOLD_PERCENT", 0.002)) # e.g., 0.002 for 0.2%
SLEEP_TIME = int(os.environ.get("SLEEP_TIME", 60*5))  # Default to 5 minutes if not set

# try to read from env var and convert to int, otherwise use default
try:
    FAST_MA_PERIOD = int(os.environ.get("FAST_MA_PERIOD", FAST_MA_PERIOD_DEFAULT))
except ValueError:
    print(f"Warning: FAST_MA_PERIOD env var is not a valid integer, using default value: {FAST_MA_PERIOD_DEFAULT}")
    FAST_MA_PERIOD = FAST_MA_PERIOD_DEFAULT
except TypeError: # os.environ.get returns None if not set
    print(f"Warning: FAST_MA_PERIOD env var is not set, using default value: {FAST_MA_PERIOD_DEFAULT}")
    FAST_MA_PERIOD = FAST_MA_PERIOD_DEFAULT

try:
    SLOW_MA_PERIOD = int(os.environ.get("SLOW_MA_PERIOD", SLOW_MA_PERIOD_DEFAULT))
except ValueError:
    print(f"Warning: SLOW_MA_PERIOD env var is not a valid integer, using default value: {SLOW_MA_PERIOD_DEFAULT}")
    SLOW_MA_PERIOD = SLOW_MA_PERIOD_DEFAULT
except TypeError: # os.environ.get returns None if not set
    print(f"Warning: SLOW_MA_PERIOD env var is not set, using default value: {SLOW_MA_PERIOD_DEFAULT}")
    SLOW_MA_PERIOD = SLOW_MA_PERIOD_DEFAULT


def initialize_client():
    """Initialize the Binance client."""
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        print("Binance client initialized.")
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return False # Indicate initialization failure
    return True # Indicate success


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
    elif fast_ma < slow_ma and rsi_value < rsi_overbought: # Corrected condition - was duplicate of BUY condition in previous version
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
        trade_qty_usdc = trade_qty * current_price # Calculate USDC spent
        if trade_qty_usdc > usdc_balance:
            print(f"Error: Insufficient USDC balance for BUY order. Required: {trade_qty_usdc:.2f}, Available: {usdc_balance:.2f}")
            return usdc_balance, trump_balance # No balance update
        usdc_balance -= trade_qty_usdc
        trump_balance += trade_qty
    elif trade_action == "SELL":
        trade_value_usdc = trade_qty * current_price # Calculate USDC gained
        if trade_qty > trump_balance:
            print(f"Error: Insufficient TRUMP balance for SELL order. Required: {trade_qty:.2f}, Available: {trump_balance:.2f}")
            return usdc_balance, trump_balance # No balance update
        usdc_balance += trade_value_usdc
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


def fetch_binance_data(symbol: str = 'BTCUSDT', interval: str = Client.KLINE_INTERVAL_1HOUR, limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical candlestick data from Binance API.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candlestick interval (e.g., Client.KLINE_INTERVAL_1HOUR).
        limit: Number of data points to fetch.

    Returns:
        Pandas DataFrame with historical data, indexed by datetime.
    """
    try:
        client = Client() # No API key needed for public endpoints
        klines = client.get_historical_klines(symbol, interval, limit=limit)

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = pd.to_numeric(df['close']) # Ensure 'close' is numeric
        return df[['close']] # Return DataFrame with only the 'close' price
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return None


def analyze_trends(time_series_data: Union[List[Tuple], pd.DataFrame], window_size=3) -> dict:
    """
    Analyzes trends in time series data. (Currently not used in main trading loop)

    Args:
        time_series_data: Time series data as a list of tuples (timestamp, value) or Pandas DataFrame.
        window_size: Window size for trend analysis.

    Returns:
        Dictionary containing trend analysis results.
    """
    if isinstance(time_series_data, pd.DataFrame):
        df = time_series_data.copy() # Work with a copy to avoid modifying original DataFrame
    else:
        df = pd.DataFrame(time_series_data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'value': 'close'}, inplace=True) # Rename to 'close' for consistency

    # Calculate moving averages as a feature
    df['MA_short'] = df['close'].rolling(window=window_size).mean()
    df['MA_long'] = df['close'].rolling(window=window_size * 2).mean() # Longer MA for comparison
    df.dropna(inplace=True) # Remove NaN values created by moving averages

    trends = []
    if not df.empty: # Check if DataFrame is not empty after dropping NaNs
        for i in range(window_size, len(df)): # Start from window_size to have MA values
            window = df['close'][i-window_size:i]
            ma_short_window = df['MA_short'][i-window_size:i]
            ma_long_window = df['MA_long'][i-window_size:i]

            # Trend detection logic -  modified to consider moving averages (basic example)
            if ma_short_window.iloc[-1] > ma_long_window.iloc[-1] and all(ma_short_window > ma_long_window): # Short MA above Long MA - potential uptrend
                trend_type = "uptrend"
            elif ma_short_window.iloc[-1] < ma_long_window.iloc[-1] and all(ma_short_window < ma_long_window): # Short MA below Long MA - potential downtrend
                trend_type = "downtrend"
            else:
                trend_type = "sideways"

            trends.append({
                "timestamp": df.index[i],
                "value": df['close'][i],
                "trend_type": trend_type
            })

    return {"trends": trends}


def calculate_additional_features(df, fast_ma_period, slow_ma_period, rsi_period):
    """Calculate additional technical indicators."""
    df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
    df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
    df['RSI'] = calculate_rsi_pandas(df['close'], rsi_period) # Use pandas RSI
    df['Price_Change'] = df['close'].pct_change() # Percentage price change
    df.dropna(inplace=True) # Drop rows with NaN after feature calculation
    return df


def calculate_rsi_pandas(series, period=14):
    """Calculate RSI using pandas for better performance."""
    delta = series.diff().dropna()
    u = delta * 0
    d = delta * 0
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = np.abs(delta[delta < 0])
    rol_up = u.rolling(window=period).mean()
    rol_down = d.rolling(window=period).mean()
    rs = rol_up / rol_down
    return 100.0 - (100.0 / (1.0 + rs))


def prepare_training_data(df):
    """Prepare data for training the ML model. (Simplified labeling for example)"""
    df['Signal'] = 0  # 0: NEUTRAL
    # Simplified signal logic: Use only MA crossover for training data generation
    df.loc[df['MA_fast'] > df['MA_slow'], 'Signal'] = 1  # BUY if fast MA is above slow MA
    df.loc[df['MA_fast'] < df['MA_slow'], 'Signal'] = -1 # SELL if fast MA is below slow MA
    df.dropna(inplace=True) # Ensure no NaNs after signal generation
    X = df[['MA_fast', 'MA_slow', 'RSI', 'Price_Change']] # Features
    y = df['Signal'] # Target variable (BUY, SELL, NEUTRAL)

    # --- Debugging: Print signal distribution ---
    print("Signal distribution in training data:")
    print(y.value_counts())
    # --- End debugging ---
    return train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% test


def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training ML model: {e}")
        return None # Indicate model training failure


def predict_signal(model, features):
    """Predict trading signal using the trained model."""
    if model is None: # Check if model is valid
        print("Warning: ML model is not trained or failed to train. Returning NEUTRAL signal.")
        return "NEUTRAL"

    # Ensure features are in DataFrame format as expected by the model
    features_df = pd.DataFrame([features], columns=['MA_fast', 'MA_slow', 'RSI', 'Price_Change'])
    try:
        prediction = model.predict(features_df)
        # Convert prediction to trading signal string
        if prediction[0] == 1:
            return "BUY"
        elif prediction[0] == -1:
            return "SELL"
        else:
            return "NEUTRAL"
    except Exception as e:
        print(f"Error during ML model prediction: {e}")
        return "NEUTRAL" # Return NEUTRAL in case of prediction error


def signal_handler(sig, frame):
    """Handles SIGINT signal (CTRL+C)."""
    global trading_enabled
    print('\nCTRL+C detected. Gracefully exiting...')
    trading_enabled = False # Stop the trading loop
    # Perform any cleanup actions here if needed (e.g., close connections, save state)


def main():
    global initial_trump_qty, current_strategy_params, historical_performance, trading_enabled

    import signal # TRY IMPORTING SIGNAL AGAIN INSIDE MAIN - FOR TESTING

    if not initialize_client(): # Initialize client and check for success
        print("Failed to initialize Binance client. Exiting.")
        return

    signal.signal(signal.SIGINT, signal_handler) # Register signal handler for CTRL+C

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

    # --- ML Model Training ---
    print("Fetching historical data for model training...")
    historical_df = fetch_binance_data(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=10000) # Fetch TRUMPUSDC, increased limit to 10000
    model = None # Initialize model to None
    if historical_df is None or historical_df.empty:
        print("Failed to fetch historical data for training. Exiting ML training.")
        model = None # No model will be used
    else:
        historical_df = calculate_additional_features(historical_df.copy(), FAST_MA_PERIOD, SLOW_MA_PERIOD, RSI_PERIOD) # Calculate features for historical data
        X_train, X_test, y_train, y_test = prepare_training_data(historical_df)
        model = train_model(X_train, y_train) # Train model and handle potential failure
        if model is not None: # Only evaluate and print accuracy if model training was successful
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
        else:
            print("ML model training failed. Trading will proceed without ML.")


    iteration = 0


    while trading_enabled: # Run as long as trading is enabled (can be stopped by CTRL+C)
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        current_price = fetch_current_price(SYMBOL)
        if not current_price:
            time.sleep(60) # Wait for 1 minute before retrying
            continue

        prices_history.append(current_price) # Append current price to history

        if len(prices_history) > SLOW_MA_PERIOD: # Only trade after enough data
            ml_signal = "NEUTRAL" # Default ML signal if model is not available or prediction fails
            if model is not None: # Check if model is trained before using it
                # Create DataFrame for current features
                current_df = pd.DataFrame({'close': prices_history[-SLOW_MA_PERIOD:]}) # Use last SLOW_MA_PERIOD prices
                current_df = calculate_additional_features(current_df, current_strategy_params["FAST_MA_PERIOD"], current_strategy_params["SLOW_MA_PERIOD"], current_strategy_params["RSI_PERIOD"])
                current_features = current_df.iloc[[-1]][['MA_fast', 'MA_slow', 'RSI', 'Price_Change']].to_dict('records')[0] # Get last row features as dict

                ml_signal = predict_signal(model, current_features) # Predict signal using ML model


            rule_based_signal = generate_trading_signal( # Original rule-based signal
                prices_history,
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"],
                current_strategy_params["RSI_OVERBOUGHT"],
                current_strategy_params["RSI_OVERSOLD"]
            )

            # --- Signal Combination Logic (Example - can be improved) ---
            if ml_signal != "NEUTRAL": # Prioritize ML signal if it's not neutral
                signal = ml_signal
                print(f"ML Signal: {ml_signal}, Rule-Based Signal: {rule_based_signal} - Using ML Signal")
            else:
                signal = rule_based_signal # Fallback to rule-based signal if ML is neutral
                print(f"ML Signal: {ml_signal}, Rule-Based Signal: {rule_based_signal} - Using Rule-Based Signal")


            print(f"Current Price: {current_price}, Combined Signal: {signal}")

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