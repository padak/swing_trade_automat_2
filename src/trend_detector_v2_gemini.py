import signal # For signal handling
import os
import time
from binance.client import Client
from binance.enums import *
import numpy as np
import pandas as pd
import pandas_ta as ta  # for technical indicators
from typing import List, Tuple, Union
from sklearn.linear_model import LogisticRegression # Example ML model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv # Import the csv module for logging
from joblib import dump, load # Import joblib for saving/loading the model

print("Signal module imported successfully:", 'signal' in globals()) # Check if signal is imported
print("TA-Lib module imported successfully:", 'ta' in globals())

# Documentation: https://python-binance.readthedocs.io/en/latest/

# --- Configuration ---
API_KEY = os.environ.get('binance_api')
API_SECRET = os.environ.get('binance_secret')
SYMBOL_raw = os.environ.get("SYMBOL", "TRUMPUSDC")  # Default to "TRUMPUSDC" if not set (still with space for now, corrected below)
SYMBOL = SYMBOL_raw.strip() # Remove leading/trailing whitespace, important for API calls
INITIAL_USDC = 500
INITIAL_TRUMP_USDC_VALUE = 500
MIN_TRADE_USDC = 1.2

# --- Strategy Parameters (Initial & Adjustable) ---
FAST_MA_PERIOD_DEFAULT = 12
SLOW_MA_PERIOD_DEFAULT = 26
RSI_PERIOD_DEFAULT = 14
RSI_OVERBOUGHT_DEFAULT = 70
RSI_OVERSOLD_DEFAULT = 30
MACD_FAST_PERIOD_DEFAULT = 12
MACD_SLOW_PERIOD_DEFAULT = 26
MACD_SIGNAL_PERIOD_DEFAULT = 9
BOLLINGER_BAND_PERIOD_DEFAULT = 20
BOLLINGER_BAND_STD_DEFAULT = 2
STOP_LOSS_PERCENT_DEFAULT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT_DEFAULT = 0.10 # 10% take profit
TRADE_RISK_PERCENT_DEFAULT = 0.01 # Risk 1% of portfolio per trade

# --- Global Variables ---
client = None
initial_trump_qty = 0.0
current_strategy_params = {}
historical_performance = []
trading_enabled = True # Flag for trading loop - initialize here
trade_history = [] # List to store trade details for performance reporting
last_performance_iteration = 0 # To track when to reset trade history
prices_history = pd.DataFrame(columns=['timestamp', 'close']) # DataFrame for price history

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
        return True
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return False # Indicate initialization failure


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
    """
    Generates a trading signal based on Moving Averages and RSI.
    Returns: signal, trend, ma_fast, ma_slow
    """
    df = pd.DataFrame({'close': prices})
    df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
    df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
    df['RSI'] = calculate_rsi_pandas(df['close'], rsi_period)

    signal = "NEUTRAL"
    trend = "NEUTRAL" # Initialize trend
    ma_fast_value = df['MA_fast'].iloc[-1] if not df['MA_fast'].empty else np.nan # Get last MA_fast value, handle empty case
    ma_slow_value = df['MA_slow'].iloc[-1] if not df['MA_slow'].empty else np.nan # Get last MA_slow value, handle empty case

    last_row = df.iloc[-1]

    if last_row['MA_fast'] > last_row['MA_slow']:
        trend = "UPTREND"
    elif last_row['MA_fast'] < last_row['MA_slow']:
        trend = "DOWNTREND"
    else:
        trend = "NEUTRAL"

    if trend == "UPTREND" and last_row['RSI'] < rsi_oversold:
        signal = "BUY"
    elif trend == "DOWNTREND" and last_row['RSI'] > rsi_overbought:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    print(f"Debug in generate_trading_signal: MA_Fast value: {ma_fast_value}, MA_Slow value: {ma_slow_value}") # Debug print - ADDED
    return signal, trend, ma_fast_value, ma_slow_value # Return signal, trend, and MA values


def execute_trade(signal, current_price, usdc_balance, trump_balance):
    """Simulate trade execution."""
    trade_qty = 0.0
    trade_details = {} # Dictionary to store trade details

    if signal == "BUY":
        trade_qty_usdc = usdc_balance * 0.99 # Use 99% of available USDC
        if trade_qty_usdc > MIN_TRADE_USDC:
            trade_qty = trade_qty_usdc / current_price
            print(f"BUY {SYMBOL}: Quantity: {trade_qty:.2f}, Price: {current_price}")
            trade_details = {
                "action": "BUY",
                "price": current_price,
                "quantity": trade_qty,
                "timestamp": time.time()
            }
            return "BUY", trade_qty, trade_details
        else:
            print(f"BUY signal, but trade quantity below minimum. USDC balance: {usdc_balance}")
            return "NEUTRAL", 0.0, trade_details # Return empty trade_details

    elif signal == "SELL":
        trade_qty = trump_balance
        if trade_qty * current_price > MIN_TRADE_USDC:
            print(f"SELL {SYMBOL}: Quantity: {trade_qty:.2f}, Price: {current_price}")
            trade_details = {
                "action": "SELL",
                "price": current_price,
                "quantity": trade_qty,
                "timestamp": time.time()
            }
            return "SELL", trade_qty, trade_details
        else:
            print(f"SELL signal, but trade value below minimum. TRUMP balance value: {trump_balance * current_price}")
            return "NEUTRAL", 0.0, trade_details # Return empty trade_details
    return "NEUTRAL", 0.0, trade_details # Return empty trade_details


def update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details):
    """Update USDC and TRUMP balances based on trade execution and record trade details."""
    if trade_action == "BUY":
        trade_qty_usdc = trade_qty * current_price # Calculate USDC spent
        if trade_qty_usdc > usdc_balance:
            print(f"Error: Insufficient USDC balance for BUY order. Required: {trade_qty_usdc:.2f}, Available: {usdc_balance:.2f}")
            return usdc_balance, trump_balance # No balance update
        usdc_balance -= trade_qty_usdc
        trump_balance += trade_qty
        trade_details["usdc_spent"] = trade_qty_usdc # Record USDC spent
    elif trade_action == "SELL":
        trade_value_usdc = trade_qty * current_price # Calculate USDC gained
        if trade_qty > trump_balance:
            print(f"Error: Insufficient TRUMP balance for SELL order. Required: {trade_qty:.2f}, Available: {trump_balance:.2f}")
            return usdc_balance, trump_balance # No balance update
        usdc_balance += trade_value_usdc
        trump_balance -= trade_qty
        trade_details["usdc_gained"] = trade_value_usdc # Record USDC gained

    if trade_details: # Only add to history if trade was executed (trade_details is not empty)
        trade_history.append(trade_details)
    return usdc_balance, trump_balance


def evaluate_performance(initial_portfolio_value, current_portfolio_value):
    """Evaluate performance and calculate profit/loss percentage."""
    profit_loss = current_portfolio_value - initial_portfolio_value
    profit_loss_percent = (profit_loss / initial_portfolio_value) * 100
    return profit_loss_percent


def adjust_strategy_parameters(average_performance, current_params):
    """Adjusts strategy parameters based on recent performance."""
    print(f"\n--- Parameter Adjustment ---")
    print(f"Current RSI Overbought: {current_params['RSI_OVERBOUGHT']}, Oversold: {current_params['RSI_OVERSOLD']}") # Debug: Before adjustment
    print(f"Debug: Full params BEFORE adjustment logic: {current_params}") # Debug: Full params before logic

    if average_performance < -0.5: # Poor performance threshold
        print("Poor performance, loosening parameters slightly.")
        if current_params['RSI_OVERBOUGHT'] < 80:
            current_params['RSI_OVERBOUGHT'] += 2
            print("Loosening RSI parameters.")
        if current_params['RSI_OVERSOLD'] > 20:
            current_params['RSI_OVERSOLD'] -= 2
            print("Loosening RSI parameters.")
        print(f"Adjusted strategy parameters: RSI Overbought={current_params['RSI_OVERBOUGHT']}, RSI Oversold={current_params['RSI_OVERSOLD']}") # Debug: After loosening
        print(f"Debug: Full params AFTER loosening: {current_params}") # Debug: Full params after loosening

    elif average_performance > 0.5: # Good performance threshold
        print("Good performance, tightening parameters slightly.")
        if current_params['RSI_OVERBOUGHT'] > 60:
            current_params['RSI_OVERBOUGHT'] -= 2
            print("Tightening RSI parameters.")
        if current_params['RSI_OVERSOLD'] < 40:
            current_params['RSI_OVERSOLD'] += 2
            print("Tightening RSI parameters.")
        print(f"Adjusted strategy parameters: RSI Overbought={current_params['RSI_OVERBOUGHT']}, RSI Oversold={current_params['RSI_OVERSOLD']}") # Debug: After tightening
        print(f"Debug: Full params AFTER tightening: {current_params}") # Debug: Full params after tightening
    else:
        print("Performance within acceptable range, no parameter adjustment needed.")
        print(f"Debug: Full params - NO adjustment: {current_params}") # Debug: Full params - no adjustment

    print(f"Debug: Parameters after adjustment: {current_params}") # Debug: Full params after adjustment
    print(f"Debug: Keys after adjustment: {current_params.keys()}") # Debug: Keys after adjustment
    return current_params


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
    global trading_enabled, model # Make 'model' global so we can access it
    print('\nCTRL+C detected. Gracefully exiting...')
    trading_enabled = False # Stop the trading loop
    # Perform any cleanup actions here if needed (e.g., close connections, save state)

    # Save the trained model before exiting
    if model: # Check if model is trained before saving
        print("Saving trained ML model...")
        try:
            dump(model, 'trading_model.joblib') # Save model to a file
            print("ML model saved successfully to trading_model.joblib")
        except Exception as e:
            print(f"Error saving ML model: {e}")


def log_data(log_file_path, log_message):
    """Logs trading data to a CSV file."""
    log_file_exists = os.path.isfile(log_file_path)
    log_header = ["Timestamp", "Iteration", "Price", "Signal", "Trend", "USDC_Balance", "TRUMP_Balance", "Portfolio_Value", "P/L_Percent", "Trade_Action", "Trade_Quantity", "Trade_Price", "RSI_Overbought", "RSI_Oversold", "MA_Fast", "MA_Slow"]
    try:
        with open(log_file_path, mode='a', newline='') as logfile:
            log_writer = csv.DictWriter(logfile, fieldnames=log_header) # Use DictWriter
            if not log_file_exists:
                log_writer.writeheader() # Write header using DictWriter
            log_writer.writerow(log_message) # Write dictionary directly
            print(f"Debug in log_data: log_message = {log_message}") # Debug print in log_data
    except Exception as e:
        print(f"Logging error: {e}")


def prepare_log_message(current_time_str, iteration, current_price, signal, trend, usdc_balance, trump_balance, current_portfolio_value, performance_percent, trade_action, trade_qty, trade_details, current_strategy_params, ma_fast, ma_slow):
    """Prepares the log message as a dictionary."""
    try: # Add try-except block
        log_message_dict = {
            "Timestamp": current_time_str,
            "Iteration": iteration,
            "Price": current_price,
            "Signal": signal,
            "Trend": trend,
            "USDC_Balance": f"{usdc_balance:.2f}",
            "TRUMP_Balance": f"{trump_balance:.2f}",
            "Portfolio_Value": f"{current_portfolio_value:.2f}",
            "P/L_Percent": f"{performance_percent:.2f}",
            "Trade_Action": trade_action,
            "Trade_Quantity": trade_qty,
            "Trade_Price": trade_details.get("price", ""),
            "RSI_Overbought": current_strategy_params.get("RSI_OVERBOUGHT", "KEY_ERROR"),
            "RSI_Oversold": current_strategy_params.get("RSI_Oversold", "KEY_ERROR"),
            "MA_Fast": ma_fast,
            "MA_Slow": ma_slow,
        }
        print(f"Debug INSIDE prepare_log_message - current_strategy_params keys: {log_message_dict.keys()}") # Debug: Keys in prepare_log_message - MOVED
        print(f"Debug INSIDE prepare_log_message - current_strategy_params: {log_message_dict}") # Debug: Full params in prepare_log_message - MOVED
        return log_message_dict
    except Exception as e: # Catch any exception
        print(f"Error in prepare_log_message: {e}") # Log the error
        return {} # Return an empty dictionary in case of error


def main():
    global initial_trump_qty, current_strategy_params, historical_performance, trading_enabled, trade_history, last_performance_iteration, prices_history, model # Make 'model' global

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
    initial_portfolio_value = INITIAL_USDC + INITIAL_TRUMP_USDC_VALUE
    benchmark_portfolio_value_history = [] # NEW: History for benchmark portfolio value - ADDED HERE

    initial_strategy_params = load_strategy_params()
    current_strategy_params = initial_strategy_params.copy() # Use .copy() to avoid modifying initial params
    print(f"Debug: Loaded strategy parameters: {current_strategy_params}")
    print(f"Debug: id(current_strategy_params) after load = {id(current_strategy_params)}") # Debug: ID after load
    print(f"Debug: current_strategy_params after load_strategy_params: {current_strategy_params}")
    print(f"Initial strategy parameters: {initial_strategy_params}")

    # --- ML Model Training ---
    print("Fetching historical data for model training...")
    historical_df = fetch_binance_data(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=10000) # Fetch TRUMPUSDC, increased limit to 10000
    model = None # Initialize model to None

    # Try to load pre-trained model
    print("Checking for saved ML model...")
    try:
        model = load('trading_model.joblib') # Load model if file exists
        print("Pre-trained ML model loaded successfully from trading_model.joblib")
    except FileNotFoundError:
        print("No saved ML model found. Training new model.")
        if historical_df is None or historical_df.empty:
            print("Failed to fetch historical data for training. Exiting ML training.")
            model = None # No model will be used
        else:
            historical_df = calculate_additional_features(
                historical_df.copy(),
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"]
            ) # Calculate features for historical data
            X_train, X_test, y_train, y_test = prepare_training_data(historical_df)
            model = train_model(X_train, y_train) # Train model and handle potential failure
            if model is not None: # Only evaluate and print accuracy if model training was successful
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
            else:
                print("ML model training failed. Trading will proceed without ML.")
    except Exception as e: # Catch other potential errors during loading
        print(f"Error loading pre-trained ML model: {e}. Training new model.")
        model = None # Ensure model is None in case of loading error
        if historical_df is None or historical_df.empty:
            print("Failed to fetch historical data for training. Exiting ML training.")
            model = None # No model will be used
        else:
            historical_df = calculate_additional_features(
                historical_df.copy(),
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"]
            ) # Calculate features for historical data
            X_train, X_test, y_train, y_test = prepare_training_data(historical_df)
            model = train_model(X_train, y_train) # Train model and handle potential failure
            if model is not None: # Only evaluate and print accuracy if model training was successful
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
            else:
                print("ML model training failed. Trading will proceed without ML.")

    iteration = 0
    log_file = "trading_log.csv" # Define log file name
    # Write CSV header
    with open(log_file, mode='w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Timestamp', 'Iteration', 'Price', 'Signal', 'Trend', 'USDC_Balance', 'TRUMP_Balance', 'Portfolio_Value', 'P/L_Percent', 'Trade_Action', 'Trade_Quantity', 'Trade_Price', 'RSI_Overbought', 'RSI_Oversold', 'MA_Fast', 'MA_Slow'])


    while trading_enabled: # Run as long as trading is enabled (can be stopped by CTRL+C)
        iteration += 1
        timestamp_now = time.time()
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_now))
        print(f"\n--- Iteration {iteration} - {current_time_str} ---")

        current_price = fetch_current_price(SYMBOL)
        if not current_price:
            time.sleep(60) # Wait for 1 minute before retrying
            continue

        prices_history.append(current_price) # Append current price to history

        signal = "NEUTRAL" # Initialize signal
        trend = "NEUTRAL" # Initialize trend
        ma_fast = np.nan    # Initialize MA_fast
        ma_slow = np.nan    # Initialize MA_slow
        trade_action = "NEUTRAL" # Initialize trade_action - ADDED
        trade_qty = 0.0 # Initialize trade_qty - ADDED
        trade_details = {} # Initialize trade_details - ADDED

        if len(prices_history) > current_strategy_params["SLOW_MA_PERIOD"]: # Ensure enough data for indicators
            signal, trend, ma_fast, ma_slow = generate_trading_signal(prices_history, current_strategy_params["FAST_MA_PERIOD"], current_strategy_params["SLOW_MA_PERIOD"], current_strategy_params["RSI_PERIOD"], current_strategy_params["RSI_OVERBOUGHT"], current_strategy_params["RSI_OVERSOLD"])
            portfolio_value = usdc_balance + (trump_balance * current_price) # Portfolio value before trade
            print(f"Current Price: {current_price:.2f}, Signal: {signal}, Portfolio Value: {portfolio_value:.2f} USDC")

            trade_action, trade_qty, trade_details = execute_trade(signal, current_price, usdc_balance, trump_balance) # Capture trade_details
            if trade_action != "NEUTRAL":
                usdc_balance, trump_balance = update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details) # Pass trade_details

        current_portfolio_value = usdc_balance + (trump_balance * current_price)
        performance_percent = evaluate_performance(initial_portfolio_value, current_portfolio_value)

        print(f"USDC Balance: {usdc_balance:.2f}, TRUMP Balance: {trump_balance:.2f}, Portfolio Value: {current_portfolio_value:.2f} USDC, P/L: {performance_percent:.2f}%  |  Current Price: {current_price:.2f}")

        historical_performance.append(performance_percent) # Store performance history

        # --- Benchmark Portfolio Calculation ---
        benchmark_portfolio_value = INITIAL_USDC + (initial_trump_qty * current_price) # Calculate benchmark value
        benchmark_portfolio_value_history.append(benchmark_portfolio_value) # Store benchmark value - ADDED HERE

        # Log data for each iteration
        log_message = prepare_log_message(current_time_str, iteration, current_price, signal, trend, usdc_balance, trump_balance, current_portfolio_value, performance_percent, trade_action, trade_qty, trade_details, current_strategy_params, ma_fast, ma_slow)
        log_data(log_file, log_message)

        if iteration % 10 == 0: # Adjust parameters every 10 iterations
            if len(historical_performance) >= 10: # Adjust based on last 10 iterations average performance
                avg_performance = np.mean(historical_performance[-10:])
                print(f"\n--- Performance Report for last 10 iterations ---")
                print(f"Average Performance: {avg_performance:.2f}%")

                # --- Benchmark Performance Calculation ---
                if len(benchmark_portfolio_value_history) >= 11: # CHECK FOR >= 11 HERE
                    initial_benchmark_value_period = benchmark_portfolio_value_history[-11] # Value 10 iterations ago
                    current_benchmark_value = benchmark_portfolio_value_history[-1] # Current benchmark value
                    benchmark_performance_percent = ((current_benchmark_value - initial_benchmark_value_period) / initial_benchmark_value_period) * 100 # Benchmark performance over last 10 iterations
                else:
                    benchmark_performance_percent = 0.0 # Not enough data yet

                print(f"Benchmark Performance (last 10 iters): {benchmark_performance_percent:.2f}%") # Print benchmark performance - ADDED HERE
                performance_vs_benchmark = avg_performance - benchmark_performance_percent # Calculate difference - ADDED HERE
                print(f"Strategy vs. Benchmark: {performance_vs_benchmark:.2f}%") # Print difference - ADDED HERE


                # --- Trade Summary ---
                recent_trades = trade_history[last_performance_iteration:] # Trades since last report
                buy_orders = [trade for trade in recent_trades if trade["action"] == "BUY"]
                sell_orders = [trade for trade in recent_trades if trade["action"] == "SELL"]

                buy_value = sum([trade.get("usdc_spent", 0) for trade in buy_orders]) # Total spent on buys
                sell_value = sum([trade.get("usdc_gained", 0) for trade in sell_orders]) # Total gained from sells
                net_profit_trades = sell_value - buy_value # Net profit from trades

                num_buy_orders = len(buy_orders)
                num_sell_orders = len(sell_orders)
                total_trades = num_buy_orders + num_sell_orders

                print(f"  Total Trades: {total_trades}")
                print(f"  BUY Orders: {num_buy_orders}")
                print(f"  SELL Orders: {num_sell_orders}")
                print(f"  Net Profit from Trades (approx): {net_profit_trades:.2f} USDC")

                current_strategy_params = adjust_strategy_parameters(avg_performance, current_strategy_params)

                print(f"Debug BEFORE PARAMS_ADJUST LOG - current_strategy_params keys: {current_strategy_params.keys()}") # Debug: Keys before params adjust log
                print(f"Debug BEFORE PARAMS_ADJUST LOG - current_strategy_params: {current_strategy_params}") # Debug: Full params before params adjust log
                print(f"Debug: id(current_strategy_params) before params adjust log = {id(current_strategy_params)}") # Debug: ID before params adjust log
                # Log parameter adjustments
                log_message_params_adjust = [current_time_str, iteration, "-", "PARAMS_ADJUST", "-", "-", "-", "-", "-", "-", "-", current_strategy_params.get("RSI_OVERBOUGHT", "KEY_ERROR"), current_strategy_params.get("RSI_Oversold", "KEY_ERROR")] # Log parameter adjustments - USE .get()
                log_data(log_file, log_message_params_adjust)


                last_performance_iteration = len(trade_history) # Update last iteration index

        time.sleep(10) # Check price every 10 seconds


def load_strategy_params():
    """Loads strategy parameters from environment variables or defaults."""
    params = {
        'FAST_MA_PERIOD': FAST_MA_PERIOD_DEFAULT,
        'SLOW_MA_PERIOD': SLOW_MA_PERIOD_DEFAULT,
        'RSI_PERIOD': RSI_PERIOD_DEFAULT,
        'RSI_OVERBOUGHT': RSI_OVERBOUGHT_DEFAULT,
        'RSI_OVERSOLD': RSI_OVERSOLD_DEFAULT,
        'MACD_FAST_PERIOD': MACD_FAST_PERIOD_DEFAULT,
        'MACD_SLOW_PERIOD': MACD_SLOW_PERIOD_DEFAULT,
        'MACD_SIGNAL_PERIOD': MACD_SIGNAL_PERIOD_DEFAULT,
        'BOLLINGER_BAND_PERIOD': BOLLINGER_BAND_PERIOD_DEFAULT,
        'BOLLINGER_BAND_STD': BOLLINGER_BAND_STD_DEFAULT,
        'STOP_LOSS_PERCENT': STOP_LOSS_PERCENT_DEFAULT,
        'TAKE_PROFIT_PERCENT': TAKE_PROFIT_PERCENT_DEFAULT,
        'TRADE_RISK_PERCENT': TRADE_RISK_PERCENT_DEFAULT
    }
    print(f"Debug: params dictionary inside load_strategy_params: {params}") # Debug print here

    # Override defaults with environment variables if set
    for key in params:
        env_var_value = os.environ.get(key)
        if env_var_value is not None:
            try:
                params[key] = type(params[key])(env_var_value) # Attempt to cast to default type
            except ValueError:
                print(f"Warning: Invalid value '{env_var_value}' for env var '{key}', using default value: {params[key]}")
            except TypeError:
                print(f"Warning: Type error for env var '{key}', using default value: {params[key]}")

    print(f"Debug: Loaded strategy parameters: {params}") # Debug print after env var overrides
    return params


if __name__ == "__main__":
    main() 