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
FAST_MA_PERIOD_DEFAULT = 50    # changed from 12 or 20
SLOW_MA_PERIOD_DEFAULT = 200   # changed from 26 or 50
RSI_PERIOD_DEFAULT = 14
RSI_OVERBOUGHT_DEFAULT = 65    # changed from 70
RSI_OVERSOLD_DEFAULT = 35      # still 35, but reaffirmed
MACD_FAST_PERIOD_DEFAULT = 12
MACD_SLOW_PERIOD_DEFAULT = 26
MACD_SIGNAL_PERIOD_DEFAULT = 9
BOLLINGER_BAND_PERIOD_DEFAULT = 20
BOLLINGER_BAND_STD_DEFAULT = 2
STOP_LOSS_PERCENT_DEFAULT = 0.02
TAKE_PROFIT_PERCENT_DEFAULT = 0.10
TRADE_RISK_PERCENT_DEFAULT = 0.01

# Introduce a new parameter: MACD_CONFIRMATION
MACD_CONFIRMATION_DEFAULT = True  # If True, we require MACD cross for SELL downtrend signals

# Introduce a new threshold if you want to confirm a minimum price change:
MIN_PRICE_CHANGE_DEFAULT = 1.0  # from analysis recommendation (was 2.0, or not used before)

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
    print(f"Warning: FAST_MA_PERIOD not valid int, using default {FAST_MA_PERIOD_DEFAULT}")
    FAST_MA_PERIOD = FAST_MA_PERIOD_DEFAULT

try:
    SLOW_MA_PERIOD = int(os.environ.get("SLOW_MA_PERIOD", SLOW_MA_PERIOD_DEFAULT))
except ValueError:
    print(f"Warning: SLOW_MA_PERIOD not valid int, using default {SLOW_MA_PERIOD_DEFAULT}")
    SLOW_MA_PERIOD = SLOW_MA_PERIOD_DEFAULT

try:
    RSI_OVERBOUGHT = int(os.environ.get("RSI_OVERBOUGHT", RSI_OVERBOUGHT_DEFAULT))
except ValueError:
    RSI_OVERBOUGHT = RSI_OVERBOUGHT_DEFAULT

try:
    RSI_OVERSOLD = int(os.environ.get("RSI_OVERSOLD", RSI_OVERSOLD_DEFAULT))
except ValueError:
    RSI_OVERSOLD = RSI_OVERSOLD_DEFAULT

# new variables for controlling logic
MACD_CONFIRMATION = bool(os.environ.get("MACD_CONFIRMATION", MACD_CONFIRMATION_DEFAULT))
MIN_PRICE_CHANGE = float(os.environ.get("MIN_PRICE_CHANGE", MIN_PRICE_CHANGE_DEFAULT))

# Add at the top with other global variables
previous_trend_state = "NEUTRAL"  # Global variable to track previous trend

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


def fetch_ma_from_binance(symbol, interval="1m", fast_period=7, slow_period=25):
    """Fetch moving averages directly from Binance API."""
    try:
        # Fetch enough candles to calculate the slower MA
        klines = client.get_klines(symbol=symbol, interval=interval, limit=slow_period + 1)
        if not klines:
            return np.nan, np.nan

        # Extract closing prices
        closes = pd.Series([float(x[4]) for x in klines])
        
        # Calculate MAs
        ma_fast = closes.rolling(window=fast_period).mean().iloc[-1]
        ma_slow = closes.rolling(window=slow_period).mean().iloc[-1]
        
        return ma_fast, ma_slow
    except Exception as e:
        print(f"Error fetching MAs from Binance: {e}")
        return np.nan, np.nan


def generate_trading_signal(prices, fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold, 
                          macd_fast_period, macd_slow_period, macd_signal_period, previous_trend):
    """
    Generates a trading signal based on Moving Averages, RSI, and MACD, focusing on trend changes.
    Returns: signal, trend, ma_fast, ma_slow, rsi_value, macd_value, macd_signal_value
    """
    signal = "NEUTRAL"
    trend = "NEUTRAL"
    macd_value = np.nan
    macd_signal_value = np.nan
    global current_time_str

    # First try to get MA values from Binance as a baseline
    ma_fast_value, ma_slow_value = fetch_ma_from_binance(SYMBOL)

    # Try to calculate local indicators if we have enough data
    required_periods = max(slow_ma_period, macd_slow_period + macd_signal_period)
    if len(prices) > required_periods:
        try:
            df = pd.DataFrame({'close': prices['close']})
            df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
            df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
            df['RSI'] = calculate_rsi_pandas(df['close'], rsi_period)

            # Calculate MACD using pandas_ta
            try:
                macd = ta.macd(df['close'],
                              fast=macd_fast_period,
                              slow=macd_slow_period,
                              signal=macd_signal_period)

                if macd is not None and not macd.empty:
                    df['MACD_line'] = macd[f'MACD_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}']
                    df['MACD_signal'] = macd[f'MACDs_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}']
                else:
                    df['MACD_line'] = np.nan
                    df['MACD_signal'] = np.nan
            except Exception as e:
                print(f"Error calculating MACD: {e}")
                df['MACD_line'] = np.nan
                df['MACD_signal'] = np.nan

            # Get last values from local data if available
            if not df['MA_fast'].empty and not df['MA_slow'].empty:
                local_ma_fast = df['MA_fast'].iloc[-1]
                local_ma_slow = df['MA_slow'].iloc[-1]
                # Only use local values if they're not nan
                if not np.isnan(local_ma_fast) and not np.isnan(local_ma_slow):
                    ma_fast_value = local_ma_fast
                    ma_slow_value = local_ma_slow

            rsi_value = df['RSI'].iloc[-1] if not df['RSI'].empty else 50  # Default RSI if not enough data
            macd_value = df['MACD_line'].iloc[-1] if 'MACD_line' in df and not df['MACD_line'].empty else np.nan
            macd_signal_value = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df and not df['MACD_signal'].empty else np.nan

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            rsi_value = 50  # Default RSI when calculation fails
            macd_value = np.nan
            macd_signal_value = np.nan
    else:
        print(f"Not enough data for indicators. Required: {required_periods}, Available: {len(prices)}")
        rsi_value = 50  # Default RSI when not enough data

    rsi_overbought_buy = rsi_overbought # Use the general overbought parameter for BUY condition
    rsi_overbought_sell = rsi_overbought # Use the same overbought parameter for SELL condition
    rsi_oversold_level = rsi_oversold # Use the general oversold parameter

    # If we still don't have valid MA values, try Binance one more time
    if np.isnan(ma_fast_value) or np.isnan(ma_slow_value):
        ma_fast_value, ma_slow_value = fetch_ma_from_binance(SYMBOL)

    if ma_fast_value > ma_slow_value:
        trend = "UPTREND"
    elif ma_fast_value < ma_slow_value:
        trend = "DOWNTREND"
    else:
        trend = "NEUTRAL"

    # More nuanced trading signal logic
    signal = "NEUTRAL"  # Default signal
    
    if trend == "UPTREND":
        # Buy conditions in uptrend:
        # 1. RSI shows oversold condition OR
        # 2. Recent trend switch from downtrend OR
        # 3. Strong uptrend confirmation (MA fast significantly above MA slow)
        ma_difference_percent = ((ma_fast_value - ma_slow_value) / ma_slow_value) * 100
        recent_price_change = (prices['close'].iloc[-1] - prices['close'].iloc[-2]) / prices['close'].iloc[-2] * 100 if len(prices) > 1 else 0
        
        if (rsi_value < rsi_oversold_level or  # Oversold condition
            previous_trend == "DOWNTREND" or    # Trend switch
            ma_difference_percent > 0.5 or       # Strong uptrend
            abs(recent_price_change) >= MIN_PRICE_CHANGE):
            signal = "BUY"
            
    elif trend == "DOWNTREND":
        # Sell conditions in downtrend:
        # 1. RSI shows overbought condition OR
        # 2. Recent trend switch from uptrend OR
        # 3. MACD confirmation OR
        # 4. Strong downtrend confirmation
        ma_difference_percent = ((ma_slow_value - ma_fast_value) / ma_fast_value) * 100
        macd_is_bearish = (not np.isnan(macd_value) and not np.isnan(macd_signal_value) and macd_value < macd_signal_value)
        
        if ((rsi_value > rsi_overbought_sell) or          # Overbought condition
            (previous_trend == "UPTREND") or              # Trend switch
            (MACD_CONFIRMATION and macd_is_bearish) or      # MACD confirmation
            ma_difference_percent > 0.5):                  # Strong downtrend
            signal = "SELL"

    return signal, trend, ma_fast_value, ma_slow_value, rsi_value, macd_value, macd_signal_value


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
                "timestamp": time.time(),
                "entry_timestamp": time.time() # Record entry timestamp
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
                "timestamp": time.time(),
                "exit_timestamp": time.time(), # Record exit timestamp
                "profit": 0.0 # Initialize profit, will be updated in update_balance
            }
            return "SELL", trade_qty, trade_details
        else:
            print(f"SELL signal, but trade value below minimum. TRUMP balance value: {trump_balance * current_price}")
            return "NEUTRAL", 0.0, trade_details # Return empty trade_details
    return "NEUTRAL", 0.0, trade_details # Return empty trade_details


def update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details):
    """Updates balances and trade history based on trade action."""
    if trade_action == "BUY":
        usdc_spent = trade_qty * current_price
        usdc_balance -= usdc_spent
        trump_balance += trade_qty
        trade_details["usdc_spent"] = usdc_spent # Log USDC spent in trade details
        trade_details["trump_received"] = trade_qty # Log TRUMP received in trade details
        trade_history.append(trade_details) # Append trade details to history

    elif trade_action == "SELL":
        usdc_gained = trade_qty * current_price
        usdc_balance += usdc_gained
        trump_balance -= trade_qty
        trade_details["usdc_gained"] = usdc_gained # Log USDC gained in trade details
        trade_details["trump_sold"] = trade_qty # Log TRUMP sold in trade details
        buy_trade = find_corresponding_buy_trade(trade_details) # Find corresponding buy trade
        if buy_trade:
            profit = usdc_gained - buy_trade.get("usdc_spent", 0) # Calculate profit
            trade_details["profit"] = profit # Store profit in sell trade details
            buy_trade["profit"] = profit # Also update profit in the corresponding buy trade for reference
            trade_details["entry_timestamp"] = buy_trade.get("entry_timestamp") # Copy entry timestamp from buy trade
            trade_details["exit_timestamp"] = time.time() # Record exit timestamp
        else:
            trade_details["profit"] = 0.0 # If no corresponding buy trade found, set profit to 0
            trade_details["exit_timestamp"] = time.time() # Record exit timestamp

        trade_history.append(trade_details) # Append trade details to history

    return usdc_balance, trump_balance


def find_corresponding_buy_trade(sell_trade):
    """Finds the most recent corresponding BUY trade for a SELL trade (FIFO)."""
    # Iterate through trade history in reverse to find the most recent buy trade without a profit yet
    for trade in reversed(trade_history):
        if trade["action"] == "BUY" and "profit" not in trade: # Find BUY trades without profit
            return trade # Return the first such buy trade found
    return None # Return None if no corresponding buy trade is found


def evaluate_performance(initial_portfolio_value, current_portfolio_value):
    """Evaluate performance and calculate profit/loss percentage."""
    profit_loss = current_portfolio_value - initial_portfolio_value
    profit_loss_percent = (profit_loss / initial_portfolio_value) * 100
    return profit_loss_percent


def adjust_strategy_parameters(average_performance, current_params):
    """Adjusts strategy parameters based on recent performance."""
    print(f"\n--- Parameter Adjustment ---")

    if average_performance < -0.5: # Poor performance threshold
        print("Poor performance, loosening parameters slightly.")
        if current_params['RSI_OVERBOUGHT'] < 80:
            current_params['RSI_OVERBOUGHT'] += 2
        if current_params['RSI_OVERSOLD'] > 20:
            current_params['RSI_OVERSOLD'] -= 2
    elif average_performance > 0.5: # Good performance threshold
        print("Good performance, tightening parameters slightly.")
        if current_params['RSI_OVERBOUGHT'] > 60:
            current_params['RSI_OVERBOUGHT'] -= 2
        if current_params['RSI_OVERSOLD'] < 40:
            current_params['RSI_OVERSOLD'] += 2
    else:
        print("Performance within acceptable range, no parameter adjustment needed.")

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


def calculate_additional_features(df, fast_ma_period, slow_ma_period, rsi_period, macd_fast_period, macd_slow_period, macd_signal_period):
    """Calculate additional technical indicators including MACD."""
    df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
    df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
    df['RSI'] = calculate_rsi(df['close'], rsi_period)
    df['Price_Change'] = df['close'].pct_change()
    macd = ta.macd(df['close'], fast=macd_fast_period, slow=macd_slow_period, signal=macd_signal_period)
    df['MACD_line'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDh_12_26_9']
    df.dropna(inplace=True)
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
    df['Signal'] = 0
    # Simplified signal logic: Use only MA crossover for training data generation
    df.loc[df['MA_fast'] > df['MA_slow'], 'Signal'] = 1
    df.loc[df['MA_fast'] < df['MA_slow'], 'Signal'] = -1
    df.dropna(inplace=True)
    X = df[['MA_fast', 'MA_slow', 'RSI', 'Price_Change', 'MACD_line', 'MACD_signal']]
    y = df['Signal']

    # --- Debugging: Print signal distribution ---
    print("Signal distribution in training data:")
    print(y.value_counts())
    # --- End debugging ---
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training ML model: {e}")
        return None


def predict_signal(model, features):
    """Predict trading signal using the trained model."""
    if model is None:
        print("Warning: ML model is not trained or failed to train. Returning NEUTRAL signal.")
        return "NEUTRAL"

    # Ensure features are in DataFrame format as expected by the model
    features_df = pd.DataFrame([features], columns=['MA_fast', 'MA_slow', 'RSI', 'Price_Change', 'MACD_line', 'MACD_signal'])
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
        return "NEUTRAL"


def signal_handler(sig, frame):
    """Handles SIGINT signal (CTRL+C)."""
    global trading_enabled, model
    print('\nCTRL+C detected. Gracefully exiting...')
    trading_enabled = False
    # Perform any cleanup actions here if needed (e.g., close connections, save state)

    # Save the trained model before exiting
    if model:
        print("Saving trained ML model...")
        try:
            dump(model, 'trading_model.joblib')
            print("ML model saved successfully to trading_model.joblib")
        except Exception as e:
            print(f"Error saving ML model: {e}")


def log_data(log_file, log_message_dict):
    """Logs trading data to a CSV file."""
    try:
        with open(log_file, mode='a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            if isinstance(log_message_dict, dict):
                log_writer.writerow([
                    log_message_dict.get("Timestamp", ""),
                    log_message_dict.get("Iteration", ""),
                    log_message_dict.get("Price", ""),
                    log_message_dict.get("Signal", ""),
                    log_message_dict.get("Trend", ""),
                    log_message_dict.get("USDC_Balance", ""),
                    log_message_dict.get("TRUMP_Balance", ""),
                    log_message_dict.get("Portfolio_Value", ""),
                    log_message_dict.get("P/L_Percent", ""),
                    log_message_dict.get("Trade_Action", ""),
                    log_message_dict.get("Trade_Quantity", ""),
                    log_message_dict.get("Trade_Price", ""),
                    log_message_dict.get("RSI_Overbought", ""),
                    log_message_dict.get("RSI_Oversold", ""),
                    log_message_dict.get("MA_Fast", ""),
                    log_message_dict.get("MA_Slow", ""),
                    log_message_dict.get("Win_Rate", ""),
                    log_message_dict.get("Avg_Trade_Duration", ""),
                    log_message_dict.get("MACD_Line", ""),
                    log_message_dict.get("MACD_Signal", "")
                ])
            elif isinstance(log_message_dict, list):
                log_writer.writerow(log_message_dict)
            else:
                print(f"Warning: log_message is not a dict or list, but: {type(log_message_dict)}")

    except Exception as e:
        print(f"Error in log_data: {e}")


def prepare_log_message(current_time_str, iteration, current_price, signal, trend, usdc_balance, trump_balance, current_portfolio_value, performance_percent, trade_action, trade_qty, trade_details, current_strategy_params, ma_fast, ma_slow):
    """Prepares the log message as a dictionary."""
    try:
        # Format numeric values only if they're not strings or "-"
        usdc_balance_str = f"{usdc_balance:.2f}" if not isinstance(usdc_balance, str) else usdc_balance
        trump_balance_str = f"{trump_balance:.2f}" if not isinstance(trump_balance, str) else trump_balance
        portfolio_value_str = f"{current_portfolio_value:.2f}" if not isinstance(current_portfolio_value, str) else current_portfolio_value
        performance_percent_str = f"{performance_percent:.2f}" if not isinstance(performance_percent, str) else performance_percent

        # Check if trade_details is a dictionary, if not use empty string
        trade_details = trade_details if isinstance(trade_details, dict) else {}

        log_message_dict = {
            "Timestamp": current_time_str,
            "Iteration": iteration,
            "Price": current_price,
            "Signal": signal,
            "Trend": trend,
            "USDC_Balance": usdc_balance_str,
            "TRUMP_Balance": trump_balance_str,
            "Portfolio_Value": portfolio_value_str,
            "P/L_Percent": performance_percent_str,
            "Trade_Action": trade_action,
            "Trade_Quantity": trade_qty,
            "Trade_Price": trade_details.get("price", ""),
            "RSI_Overbought": current_strategy_params.get("RSI_OVERBOUGHT", "KEY_ERROR"),
            "RSI_Oversold": current_strategy_params.get("RSI_OVERSOLD", "KEY_ERROR"),
            "MA_Fast": ma_fast,
            "MA_Slow": ma_slow,
            "Win_Rate": "n/a",
            "Avg_Trade_Duration": "n/a",
            "MACD_Line": trade_details.get("macd_line", "n/a"),
            "MACD_Signal": trade_details.get("macd_signal", "n/a")
        }
        return log_message_dict
    except Exception as e:
        print(f"Error in prepare_log_message: {e}")
        # Return a safe default dictionary with all string values
        return {
            "Timestamp": str(current_time_str),
            "Iteration": str(iteration),
            "Price": str(current_price),
            "Signal": str(signal),
            "Trend": str(trend),
            "USDC_Balance": str(usdc_balance),
            "TRUMP_Balance": str(trump_balance),
            "Portfolio_Value": str(current_portfolio_value),
            "P/L_Percent": str(performance_percent),
            "Trade_Action": str(trade_action),
            "Trade_Quantity": str(trade_qty),
            "Trade_Price": "",  # Empty string for trade price when trade_details is not a dict
            "RSI_Overbought": str(current_strategy_params.get("RSI_OVERBOUGHT", "KEY_ERROR")),
            "RSI_Oversold": str(current_strategy_params.get("RSI_OVERSOLD", "KEY_ERROR")),
            "MA_Fast": str(ma_fast),
            "MA_Slow": str(ma_slow),
            "Win_Rate": "n/a",
            "Avg_Trade_Duration": "n/a",
            "MACD_Line": "n/a",  # Default value when trade_details is not a dict
            "MACD_Signal": "n/a"  # Default value when trade_details is not a dict
        }


def main():
    global initial_trump_qty, current_strategy_params, historical_performance, trading_enabled, trade_history, last_performance_iteration, prices_history, model, previous_trend_state

    import signal

    if not initialize_client():
        print("Failed to initialize Binance client. Exiting.")
        return

    signal.signal(signal.SIGINT, signal_handler)

    initial_trump_qty = calculate_initial_trump_quantity(INITIAL_TRUMP_USDC_VALUE)

    if initial_trump_qty == 0.0:
        print("Could not fetch initial TRUMP price. Exiting.")
        return

    usdc_balance = INITIAL_USDC
    trump_balance = initial_trump_qty
    prices_history = pd.DataFrame(columns=['timestamp', 'close'])
    initial_portfolio_value = INITIAL_USDC + INITIAL_TRUMP_USDC_VALUE
    benchmark_portfolio_value_history = []

    initial_strategy_params = load_strategy_params()
    current_strategy_params = initial_strategy_params.copy()
    print(f"Debug: Loaded strategy parameters: {current_strategy_params}")
    print(f"Debug: id(current_strategy_params) after load = {id(current_strategy_params)}")
    print(f"Debug: current_strategy_params after load_strategy_params: {current_strategy_params}")
    print(f"Initial strategy parameters: {initial_strategy_params}")

    print("Fetching historical data for model training...")
    historical_df = fetch_binance_data(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=10000)
    model = None

    print("Checking for saved ML model...")
    try:
        model = load('trading_model.joblib')
        print("Pre-trained ML model loaded successfully from trading_model.joblib")
    except FileNotFoundError:
        print("No saved ML model found. Training new model.")
        if historical_df is None or historical_df.empty:
            print("Failed to fetch historical data for training. Exiting ML training.")
            model = None
        else:
            historical_df = calculate_additional_features(
                historical_df.copy(),
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"],
                current_strategy_params["MACD_FAST_PERIOD"],
                current_strategy_params["MACD_SLOW_PERIOD"],
                current_strategy_params["MACD_SIGNAL_PERIOD"]
            )
            X_train, X_test, y_train, y_test = prepare_training_data(historical_df)
            model = train_model(X_train, y_train)
            if model is not None:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
            else:
                print("ML model training failed. Trading will proceed without ML.")
    except Exception as e:
        print(f"Error loading pre-trained ML model: {e}. Training new model.")
        model = None
        if historical_df is None or historical_df.empty:
            print("Failed to fetch historical data for training. Exiting ML training.")
            model = None
        else:
            historical_df = calculate_additional_features(
                historical_df.copy(),
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"],
                current_strategy_params["MACD_FAST_PERIOD"],
                current_strategy_params["MACD_SLOW_PERIOD"],
                current_strategy_params["MACD_SIGNAL_PERIOD"]
            )
            X_train, X_test, y_train, y_test = prepare_training_data(historical_df)
            model = train_model(X_train, y_train)
            if model is not None:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
            else:
                print("ML model training failed. Trading will proceed without ML.")

    iteration = 0
    log_file = "trading_log.csv"

    # Remove the code that deletes the existing file
    # Instead, only create the file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['Timestamp', 'Iteration', 'Price', 'Signal', 'Trend', 'USDC_Balance', 'TRUMP_Balance', 
                               'Portfolio_Value', 'P/L_Percent', 'Trade_Action', 'Trade_Quantity', 'Trade_Price', 
                               'RSI_Overbought', 'RSI_Oversold', 'MA_Fast', 'MA_Slow', 'Win_Rate', 'Avg_Trade_Duration', 
                               'MACD_Line', 'MACD_Signal'])
        print(f"Created new log file: {log_file}")
    else:
        print(f"Appending to existing log file: {log_file}")

    while trading_enabled:
        iteration += 1
        timestamp_now = time.time()
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_now))
        print(f"\n--- Iteration {iteration} - {current_time_str} ---")

        signal = "NEUTRAL"
        trend = "NEUTRAL"
        trade_action = "NEUTRAL"
        trade_qty = 0.0
        trade_details = {}
        ma_fast = np.nan
        ma_slow = np.nan
        rsi_value = np.nan
        macd_value = np.nan
        macd_signal_value = np.nan

        # Fetch current price and handle None case
        current_price = fetch_current_price(SYMBOL)
        if current_price is None:
            # Use last known price if available
            if not prices_history.empty:
                current_price = prices_history['close'].iloc[-1]
                print(f"Price fetch failed. Using last known price: {current_price}")
            else:
                print("Price fetch failed and no historical data available. Skipping iteration.")
                time.sleep(10)
                continue

        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([{'timestamp': current_time_str, 'close': current_price}])
        prices_history = pd.concat([prices_history, new_row], ignore_index=True)

        if len(prices_history) > current_strategy_params["SLOW_MA_PERIOD"]:
            signal, trend, ma_fast, ma_slow, rsi_value, macd_value, macd_signal_value = generate_trading_signal(
                prices_history,
                current_strategy_params["FAST_MA_PERIOD"],
                current_strategy_params["SLOW_MA_PERIOD"],
                current_strategy_params["RSI_PERIOD"],
                current_strategy_params["RSI_OVERBOUGHT"],
                current_strategy_params["RSI_OVERSOLD"],
                current_strategy_params["MACD_FAST_PERIOD"],
                current_strategy_params["MACD_SLOW_PERIOD"],
                current_strategy_params["MACD_SIGNAL_PERIOD"],
                previous_trend_state  # Pass the stored previous trend
            )
            
            # Store the current trend for the next iteration
            previous_trend_state = trend

            portfolio_value = usdc_balance + (trump_balance * current_price)
            print(f"Current Price: {current_price:.2f}, Signal: {signal}, Portfolio Value: {portfolio_value:.2f} USDC, RSI: {rsi_value:.2f}, MACD: {macd_value:.4f}, MACD Signal: {macd_signal_value:.4f}")

            trade_action, trade_qty, trade_details = execute_trade(signal, current_price, usdc_balance, trump_balance)
            if trade_action != "NEUTRAL":
                usdc_balance, trump_balance = update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details)

            trade_details["rsi_value"] = rsi_value
            trade_details["macd_line"] = macd_value
            trade_details["macd_signal"] = macd_signal_value

        current_portfolio_value = usdc_balance + (trump_balance * current_price)
        performance_percent = evaluate_performance(initial_portfolio_value, current_portfolio_value)

        print(f"USDC Balance: {usdc_balance:.2f}, TRUMP Balance: {trump_balance:.2f}, Portfolio Value: {current_portfolio_value:.2f} USDC, P/L: {performance_percent:.2f}%  |  Current Price: {current_price:.2f}")

        historical_performance.append(performance_percent)

        benchmark_portfolio_value = INITIAL_USDC + (initial_trump_qty * current_price)
        benchmark_portfolio_value_history.append(benchmark_portfolio_value)

        log_message = prepare_log_message(current_time_str, iteration, current_price, signal, trend, usdc_balance, trump_balance, current_portfolio_value, performance_percent, trade_action, trade_qty, trade_details, current_strategy_params, ma_fast, ma_slow)
        log_data(log_file, log_message)

        if iteration % 10 == 0:
            if len(historical_performance) >= 10:
                avg_performance = np.mean(historical_performance[-10:])
                print(f"\n--- Performance Report for last 10 iterations ---")
                print(f"Average Performance: {avg_performance:.2f}%")

                if len(benchmark_portfolio_value_history) >= 11:
                    initial_benchmark_value_period = benchmark_portfolio_value_history[-11]
                    current_benchmark_value = benchmark_portfolio_value_history[-1]
                    benchmark_performance_percent = ((current_benchmark_value - initial_benchmark_value_period) / initial_benchmark_value_period) * 100
                else:
                    benchmark_performance_percent = 0.0

                print(f"Benchmark Performance (last 10 iters): {benchmark_performance_percent:.2f}%")
                performance_vs_benchmark = avg_performance - benchmark_performance_percent
                print(f"Strategy vs. Benchmark: {performance_vs_benchmark:.2f}%")

                recent_trades = trade_history[last_performance_iteration:]
                buy_orders = [trade for trade in recent_trades if trade["action"] == "BUY"]
                sell_orders = [trade for trade in recent_trades if trade["action"] == "SELL"]

                buy_value = sum([trade.get("usdc_spent", 0) for trade in buy_orders])
                sell_value = sum([trade.get("usdc_gained", 0) for trade in sell_orders])
                net_profit_trades = sell_value - buy_value

                num_buy_orders = len(buy_orders)
                num_sell_orders = len(sell_orders)
                total_trades = num_buy_orders + num_sell_orders

                print(f"  Total Trades: {total_trades}")
                print(f"  BUY Orders: {num_buy_orders}")
                print(f"  SELL Orders: {num_sell_orders}")
                print(f"  Net Profit from Trades (approx): {net_profit_trades:.2f} USDC")

                closed_trades = [trade for trade in recent_trades if "profit" in trade]
                winning_trades = [trade for trade in closed_trades if trade.get("profit", 0) > 0]
                num_closed_trades = len(closed_trades)
                num_winning_trades = len(winning_trades)
                win_rate = (num_winning_trades / num_closed_trades) * 100 if num_closed_trades > 0 else 0.0

                trade_durations = [trade.get("exit_timestamp", 0) - trade.get("entry_timestamp", 0) for trade in closed_trades if "entry_timestamp" in trade and "exit_timestamp" in trade and trade.get("entry_timestamp", 0) > 0]
                avg_trade_duration_sec = np.mean(trade_durations) if trade_durations else 0
                avg_trade_duration_min = avg_trade_duration_sec / 60

                print(f"  Win Rate: {win_rate:.2f}%")
                print(f"  Avg Trade Duration: {avg_trade_duration_min:.2f} minutes")

                current_strategy_params = adjust_strategy_parameters(avg_performance, current_strategy_params)

                log_message_params_adjust = [current_time_str, iteration, "-", "PARAMS_ADJUST", "-", "-", "-", "-", "-", "-", "-", current_strategy_params.get("RSI_OVERBOUGHT", "KEY_ERROR"), current_strategy_params.get("RSI_OVERSOLD", "KEY_ERROR"), win_rate, avg_trade_duration_min]
                log_data(log_file, log_message_params_adjust)

                log_message_performance_report = prepare_log_message(current_time_str, iteration, "-", "REPORT", "-", "-", "-", "-", avg_performance, "-", "-", "-", current_strategy_params, "n/a", "n/a")
                log_message_performance_report["Win_Rate"] = f"{win_rate:.2f}%"
                log_message_performance_report["Avg_Trade_Duration"] = f"{avg_trade_duration_min:.2f} min"
                log_data(log_file, log_message_performance_report)

                last_performance_iteration = len(trade_history)

        time.sleep(10)


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
    print(f"Debug: params dictionary inside load_strategy_params: {params}")

    for key in params:
        env_var_value = os.environ.get(key)
        if env_var_value is not None:
            try:
                params[key] = type(params[key])(env_var_value)
            except ValueError:
                print(f"Warning: Invalid value '{env_var_value}' for env var '{key}', using default value: {params[key]}")
            except TypeError:
                print(f"Warning: Type error for env var '{key}', using default value: {params[key]}")

    return params


if __name__ == "__main__":
    main() 