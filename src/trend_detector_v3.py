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
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

print("Signal module imported successfully:", 'signal' in globals()) # Check if signal is imported
print("TA-Lib module imported successfully:", 'ta' in globals())

# --- Configuration ---
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

# Debug prints
print("Debug: BINANCE_API_KEY present:", bool(API_KEY))
print("Debug: BINANCE_API_SECRET present:", bool(API_SECRET))

SYMBOL_raw = os.environ.get("SYMBOL", "TRUMPUSDC")  # Default to "TRUMPUSDC" if not set
SYMBOL = SYMBOL_raw.strip() # Remove leading/trailing whitespace
INITIAL_USDC = 500
INITIAL_TRUMP_USDC_VALUE = 500
MIN_TRADE_USDC = 1.2

# --- Strategy Parameters (Initial & Adjustable) ---
FAST_MA_PERIOD_DEFAULT = 10
SLOW_MA_PERIOD_DEFAULT = 30
RSI_PERIOD_DEFAULT = 14
RSI_OVERBOUGHT_DEFAULT = 70
RSI_OVERSOLD_DEFAULT = 32  # Optimized based on simulations
MACD_FAST_PERIOD_DEFAULT = 12
MACD_SLOW_PERIOD_DEFAULT = 26
MACD_SIGNAL_PERIOD_DEFAULT = 9
BOLLINGER_BAND_PERIOD_DEFAULT = 20
BOLLINGER_BAND_STD_DEFAULT = 2
STOP_LOSS_PERCENT_DEFAULT = 0.02
TAKE_PROFIT_PERCENT_DEFAULT = 0.10
TRADE_RISK_PERCENT_DEFAULT = 0.01

# New parameters based on simulation improvements
MIN_HOLD_TIME_MINUTES_DEFAULT = 5  # Minimum hold time to avoid quick reversals
MIN_PRICE_CHANGE_DEFAULT = 1.2  # Balanced value between v2 (1.5) and v3 (1.0)
MACD_CONFIRMATION_DEFAULT = True
VOLUME_THRESHOLD_DEFAULT = 1.5  # Minimum volume multiplier compared to average
TREND_STRENGTH_THRESHOLD_DEFAULT = 0.6  # Required strength for trend confirmation

# --- Global Variables ---
client = None
initial_trump_qty = 0.0
current_strategy_params = {}
historical_performance = []
trading_enabled = True
trade_history = []
last_performance_iteration = 0
prices_history = pd.DataFrame(columns=['timestamp', 'close', 'volume'])
previous_trend_state = "NEUTRAL"

# Add these constants for logging and model storage
MODEL_DIR = "model"
LOGS_DIR = "logs"
MODEL_PATH = os.path.join(MODEL_DIR, "trading_model_v3.joblib")
TRADING_LOG_PATH = os.path.join(LOGS_DIR, "trading_log_v3.csv")

def calculate_trend_strength(df, window=20):
    """Calculate trend strength based on price movement consistency."""
    price_changes = df['close'].pct_change()
    positive_moves = (price_changes > 0).rolling(window=window).mean()
    negative_moves = (price_changes < 0).rolling(window=window).mean()
    return abs(positive_moves - negative_moves)

def calculate_volume_ratio(df, window=20):
    """Calculate volume ratio compared to moving average."""
    volume_ma = df['volume'].rolling(window=window).mean()
    return df['volume'] / volume_ma

def check_minimum_hold_time(trade_time, min_hold_time_minutes):
    """Check if minimum hold time has elapsed."""
    return (time.time() - trade_time) >= (min_hold_time_minutes * 60)

def generate_trading_signal(prices, fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold, 
                          macd_fast_period, macd_slow_period, macd_signal_period, previous_trend,
                          min_hold_time_minutes, volume_threshold, trend_strength_threshold):
    """
    Enhanced trading signal generation with additional confirmations.
    """
    trade_signal = "NEUTRAL"
    trend = "NEUTRAL"
    macd_value = np.nan
    macd_signal_value = np.nan

    try:
        df = pd.DataFrame({'close': prices['close'], 'volume': prices['volume']})
        
        # Calculate basic indicators
        df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
        df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
        df['RSI'] = calculate_rsi_pandas(df['close'], rsi_period)
        
        # Calculate MACD
        macd = ta.macd(df['close'],
                      fast=macd_fast_period,
                      slow=macd_slow_period,
                      signal=macd_signal_period)
        
        if macd is not None and not macd.empty:
            df['MACD_line'] = macd[f'MACD_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}']
            df['MACD_signal'] = macd[f'MACDs_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}']
        
        # Calculate additional confirmations
        df['trend_strength'] = calculate_trend_strength(df)
        df['volume_ratio'] = calculate_volume_ratio(df)
        
        # Get latest values
        ma_fast_value = df['MA_fast'].iloc[-1]
        ma_slow_value = df['MA_slow'].iloc[-1]
        rsi_value = df['RSI'].iloc[-1]
        macd_value = df['MACD_line'].iloc[-1]
        macd_signal_value = df['MACD_signal'].iloc[-1]
        trend_strength = df['trend_strength'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Determine trend
        if ma_fast_value > ma_slow_value:
            trend = "UPTREND"
        elif ma_fast_value < ma_slow_value:
            trend = "DOWNTREND"
        
        # Calculate momentum
        ma_difference_percent = ((ma_fast_value - ma_slow_value) / ma_slow_value) * 100
        recent_price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        
        # MACD confirmation
        macd_is_bullish = (not np.isnan(macd_value) and not np.isnan(macd_signal_value) and macd_value > macd_signal_value)
        macd_is_bearish = (not np.isnan(macd_value) and not np.isnan(macd_signal_value) and macd_value < macd_signal_value)
        
        # Enhanced trading logic with multiple confirmations
        if (
            # Buy conditions
            (trend == "DOWNTREND" and (
                (rsi_value <= rsi_oversold) or  # Oversold condition
                (previous_trend == "DOWNTREND" and trend == "UPTREND")  # Trend reversal
            )) and
            volume_ratio >= volume_threshold and  # Volume confirmation
            trend_strength >= trend_strength_threshold and  # Strong trend
            abs(recent_price_change) >= MIN_PRICE_CHANGE_DEFAULT and  # Significant price movement
            macd_is_bullish  # MACD confirmation
        ):
            trade_signal = "BUY"
        
        elif (
            # Sell conditions
            (trend == "UPTREND" and (
                (rsi_value >= rsi_overbought) or  # Overbought condition
                (previous_trend == "UPTREND" and trend == "DOWNTREND")  # Trend reversal
            )) and
            volume_ratio >= volume_threshold and  # Volume confirmation
            trend_strength >= trend_strength_threshold and  # Strong trend
            abs(recent_price_change) >= MIN_PRICE_CHANGE_DEFAULT and  # Significant price movement
            macd_is_bearish  # MACD confirmation
        ):
            trade_signal = "SELL"
        
        return trade_signal, trend, ma_fast_value, ma_slow_value, rsi_value, macd_value, macd_signal_value

    except Exception as e:
        print(f"Error in generate_trading_signal: {e}")
        return "NEUTRAL", "NEUTRAL", np.nan, np.nan, 50, np.nan, np.nan

def execute_trade(trade_signal, current_price, usdc_balance, trump_balance, last_trade_time, min_hold_time_minutes):
    """Enhanced trade execution with minimum hold time check."""
    trade_qty = 0.0
    trade_details = {}
    
    # Check minimum hold time
    if not check_minimum_hold_time(last_trade_time, min_hold_time_minutes):
        return "NEUTRAL", 0.0, {"action": "NEUTRAL", "message": "Minimum hold time not elapsed"}
    
    if trade_signal == "BUY":
        trade_qty_usdc = usdc_balance * 0.99
        if trade_qty_usdc > MIN_TRADE_USDC:
            trade_qty = trade_qty_usdc / current_price
            trade_details = {
                "action": "BUY",
                "price": current_price,
                "quantity": trade_qty,
                "timestamp": time.time(),
                "entry_timestamp": time.time()
            }
            return "BUY", trade_qty, trade_details
    
    elif trade_signal == "SELL":
        trade_qty = trump_balance
        if trade_qty * current_price > MIN_TRADE_USDC:
            trade_details = {
                "action": "SELL",
                "price": current_price,
                "quantity": trade_qty,
                "timestamp": time.time(),
                "exit_timestamp": time.time(),
                "profit": 0.0
            }
            return "SELL", trade_qty, trade_details
    
    return "NEUTRAL", 0.0, {"action": "NEUTRAL", "message": "No trade conditions met"}

def initialize_client():
    """Initialize Binance client with API credentials."""
    global client
    try:
        if not API_KEY or not API_SECRET:
            print("API credentials not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
            return False
        
        client = Client(API_KEY, API_SECRET)
        # Test API connection
        client.get_account()
        print("Successfully connected to Binance API")
        return True
    except Exception as e:
        print(f"Failed to initialize Binance client: {e}")
        return False

def fetch_current_price(symbol):
    """Fetch current price for the given symbol."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def calculate_rsi_pandas(prices, period=14):
    """Calculate RSI using pandas."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    global trading_enabled
    print('\nReceived interrupt signal. Gracefully shutting down...')
    trading_enabled = False

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
        'TRADE_RISK_PERCENT': TRADE_RISK_PERCENT_DEFAULT,
        'MIN_HOLD_TIME_MINUTES': MIN_HOLD_TIME_MINUTES_DEFAULT,
        'VOLUME_THRESHOLD': VOLUME_THRESHOLD_DEFAULT,
        'TREND_STRENGTH_THRESHOLD': TREND_STRENGTH_THRESHOLD_DEFAULT
    }

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

def fetch_historical_klines(symbol, interval="1m", limit=100):
    """
    Fetch historical klines (candlestick data) from Binance.
    interval options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def evaluate_performance(initial_portfolio_value, current_portfolio_value):
    """Evaluate performance and calculate profit/loss percentage."""
    profit_loss = current_portfolio_value - initial_portfolio_value
    profit_loss_percent = (profit_loss / initial_portfolio_value) * 100
    return profit_loss_percent

def update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details):
    """Updates balances and trade history based on trade action."""
    if trade_action == "BUY":
        usdc_spent = trade_qty * current_price
        usdc_balance -= usdc_spent
        trump_balance += trade_qty
        trade_details["usdc_spent"] = usdc_spent
        trade_details["trump_received"] = trade_qty
        trade_history.append(trade_details)

    elif trade_action == "SELL":
        usdc_gained = trade_qty * current_price
        usdc_balance += usdc_gained
        trump_balance -= trade_qty
        trade_details["usdc_gained"] = usdc_gained
        trade_details["trump_sold"] = trade_qty
        
        # Find corresponding buy trade for profit calculation
        buy_trade = None
        for trade in reversed(trade_history):
            if trade["action"] == "BUY" and "profit" not in trade:
                buy_trade = trade
                break
                
        if buy_trade:
            profit = usdc_gained - buy_trade.get("usdc_spent", 0)
            trade_details["profit"] = profit
            buy_trade["profit"] = profit
            trade_details["entry_timestamp"] = buy_trade.get("entry_timestamp")
            trade_details["exit_timestamp"] = time.time()
        else:
            trade_details["profit"] = 0.0
            trade_details["exit_timestamp"] = time.time()

        trade_history.append(trade_details)

    return usdc_balance, trump_balance

def prepare_log_message(current_time_str, iteration, current_price, signal, trend, usdc_balance, trump_balance, 
                     portfolio_value, performance_percent, trade_action, trade_qty, trade_details, strategy_params,
                     ma_fast, ma_slow):
    """Prepares the log message as a dictionary."""
    try:
        # Format numeric values
        usdc_balance_str = f"{usdc_balance:.2f}" if not isinstance(usdc_balance, str) else usdc_balance
        trump_balance_str = f"{trump_balance:.2f}" if not isinstance(trump_balance, str) else trump_balance
        portfolio_value_str = f"{portfolio_value:.2f}" if not isinstance(portfolio_value, str) else portfolio_value
        performance_percent_str = f"{performance_percent:.2f}" if not isinstance(performance_percent, str) else performance_percent

        # Check if trade_details is a dictionary
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
            "RSI_Overbought": strategy_params.get("RSI_OVERBOUGHT", ""),
            "RSI_Oversold": strategy_params.get("RSI_OVERSOLD", ""),
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
        return {}

def log_data(log_file, log_message):
    """Logs trading data to a CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Timestamp", "Iteration", "Price", "Signal", "Trend",
                    "USDC_Balance", "TRUMP_Balance", "Portfolio_Value", "P/L_Percent",
                    "Trade_Action", "Trade_Quantity", "Trade_Price",
                    "RSI_Overbought", "RSI_Oversold", "MA_Fast", "MA_Slow",
                    "Win_Rate", "Avg_Trade_Duration", "MACD_Line", "MACD_Signal"
                ])

        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if isinstance(log_message, dict):
                writer.writerow([
                    log_message.get("Timestamp", ""),
                    log_message.get("Iteration", ""),
                    log_message.get("Price", ""),
                    log_message.get("Signal", ""),
                    log_message.get("Trend", ""),
                    log_message.get("USDC_Balance", ""),
                    log_message.get("TRUMP_Balance", ""),
                    log_message.get("Portfolio_Value", ""),
                    log_message.get("P/L_Percent", ""),
                    log_message.get("Trade_Action", ""),
                    log_message.get("Trade_Quantity", ""),
                    log_message.get("Trade_Price", ""),
                    log_message.get("RSI_Overbought", ""),
                    log_message.get("RSI_Oversold", ""),
                    log_message.get("MA_Fast", ""),
                    log_message.get("MA_Slow", ""),
                    log_message.get("Win_Rate", ""),
                    log_message.get("Avg_Trade_Duration", ""),
                    log_message.get("MACD_Line", ""),
                    log_message.get("MACD_Signal", "")
                ])
    except Exception as e:
        print(f"Error in log_data: {e}")

def main():
    global initial_trump_qty, current_strategy_params, historical_performance
    global trading_enabled, trade_history, last_performance_iteration
    global prices_history, model, previous_trend_state
    
    if not initialize_client():
        print("Failed to initialize Binance client. Exiting.")
        return
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize strategy parameters
    current_strategy_params = load_strategy_params()
    print("Strategy parameters loaded:", current_strategy_params)
    
    # Create necessary directories
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Calculate initial TRUMP quantity based on current price
    current_price = fetch_current_price(SYMBOL)
    if current_price:
        initial_trump_qty = INITIAL_TRUMP_USDC_VALUE / current_price
        print(f"Initial TRUMP quantity calculated: {initial_trump_qty:.6f} at price {current_price}")
    else:
        print("Failed to fetch initial price. Using 0 as initial TRUMP quantity.")
        initial_trump_qty = 0
    
    # Initialize trading variables
    usdc_balance = INITIAL_USDC
    trump_balance = initial_trump_qty
    initial_portfolio_value = INITIAL_USDC + (initial_trump_qty * current_price if current_price else 0)
    last_trade_time = 0
    iteration = 0
    
    print("\nStarting improved trading strategy v3...")
    print(f"Initial USDC balance: {usdc_balance}")
    print(f"Initial TRUMP balance: {trump_balance}")
    print(f"Initial portfolio value: {initial_portfolio_value}")
    
    # Fetch initial historical data
    required_points = max(
        current_strategy_params["SLOW_MA_PERIOD"],
        current_strategy_params["MACD_SLOW_PERIOD"] + current_strategy_params["MACD_SIGNAL_PERIOD"]
    )
    print(f"\nFetching initial historical data (last {required_points} points)...")
    historical_df = fetch_historical_klines(SYMBOL, interval="1m", limit=required_points)
    
    if historical_df is not None:
        prices_history = historical_df[['timestamp', 'close', 'volume']]
        print(f"Successfully loaded {len(prices_history)} historical data points")
    else:
        print("Failed to fetch historical data. Starting with empty history.")
        prices_history = pd.DataFrame(columns=['timestamp', 'close', 'volume'])
    
    print("\nPress Ctrl+C to stop the bot gracefully")
    
    while trading_enabled:
        try:
            iteration += 1
            print(f"\nIteration {iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            current_price = fetch_current_price(SYMBOL)
            if current_price is None:
                print("Failed to fetch price, retrying in 10 seconds...")
                time.sleep(10)
                continue
            
            print(f"Current {SYMBOL} price: {current_price}")
            
            # Update price history with volume
            try:
                ticker = client.get_symbol_ticker(symbol=SYMBOL)
                volume = float(ticker.get('volume', 0))
                print(f"24h Volume: {volume}")
            except Exception as e:
                print(f"Error fetching volume: {e}")
                volume = 0
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([{
                'timestamp': current_time,
                'close': current_price,
                'volume': volume
            }])
            prices_history = pd.concat([prices_history, new_row], ignore_index=True)
            
            if len(prices_history) > current_strategy_params["SLOW_MA_PERIOD"]:
                print("Generating trading signals...")
                # Generate trading signal with enhanced parameters
                trade_signal, trend, ma_fast, ma_slow, rsi_value, macd_value, macd_signal_value = generate_trading_signal(
                    prices_history,
                    current_strategy_params["FAST_MA_PERIOD"],
                    current_strategy_params["SLOW_MA_PERIOD"],
                    current_strategy_params["RSI_PERIOD"],
                    current_strategy_params["RSI_OVERBOUGHT"],
                    current_strategy_params["RSI_OVERSOLD"],
                    current_strategy_params["MACD_FAST_PERIOD"],
                    current_strategy_params["MACD_SLOW_PERIOD"],
                    current_strategy_params["MACD_SIGNAL_PERIOD"],
                    previous_trend_state,
                    current_strategy_params.get("MIN_HOLD_TIME_MINUTES", MIN_HOLD_TIME_MINUTES_DEFAULT),
                    current_strategy_params.get("VOLUME_THRESHOLD", VOLUME_THRESHOLD_DEFAULT),
                    current_strategy_params.get("TREND_STRENGTH_THRESHOLD", TREND_STRENGTH_THRESHOLD_DEFAULT)
                )
                
                print(f"Signal: {trade_signal}, Trend: {trend}")
                print(f"Technical Indicators - RSI: {rsi_value:.2f}, MACD: {macd_value:.4f}, MACD Signal: {macd_signal_value:.4f}")
                
                previous_trend_state = trend
                
                # Execute trade with minimum hold time check
                trade_action, trade_qty, trade_details = execute_trade(
                    trade_signal, current_price, usdc_balance, trump_balance,
                    last_trade_time, current_strategy_params.get("MIN_HOLD_TIME_MINUTES", MIN_HOLD_TIME_MINUTES_DEFAULT)
                )
                
                if trade_action != "NEUTRAL":
                    print(f"Executing {trade_action} trade...")
                    usdc_balance, trump_balance = update_balance(
                        trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details
                    )
                    last_trade_time = time.time()
                    print(f"Trade executed - New USDC balance: {usdc_balance:.2f}, New TRUMP balance: {trump_balance:.2f}")
                
                # Log trading activity
                current_portfolio_value = usdc_balance + (trump_balance * current_price)
                performance_percent = evaluate_performance(initial_portfolio_value, current_portfolio_value)
                
                print(f"Current portfolio value: {current_portfolio_value:.2f} USDC (P/L: {performance_percent:.2f}%)")
                
                log_message = prepare_log_message(
                    current_time, iteration, current_price, trade_signal, trend,
                    usdc_balance, trump_balance, current_portfolio_value, performance_percent,
                    trade_action, trade_qty, trade_details, current_strategy_params,
                    ma_fast, ma_slow
                )
                log_data(TRADING_LOG_PATH, log_message)
            else:
                print(f"Collecting price history... ({len(prices_history)}/{current_strategy_params['SLOW_MA_PERIOD']} data points)")
            
            print(f"Waiting {10} seconds before next iteration...")
            time.sleep(10)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    main()
