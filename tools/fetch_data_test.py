import pandas as pd
from binance.client import Client
import os
import numpy as np

# --- Configuration ---
API_KEY = os.environ.get('binance_api') # You might not need API keys for public data, but it's good practice to include if your function uses it
API_SECRET = os.environ.get('binance_secret')
SYMBOL = "TRUMPUSDC" # Or you can test with other symbols like BTCUSDT, ETHUSDT
FAST_MA_PERIOD = 5
SLOW_MA_PERIOD = 20
RSI_PERIOD = 14


def fetch_binance_data(symbol: str = 'BTCUSDT', interval: str = Client.KLINE_INTERVAL_1HOUR, limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical candlestick data from Binance API.
    (Copied from trend_detector_v1_gemini.py for testing purposes)
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


def calculate_rsi_pandas(series, period=14):
    """Calculate RSI using pandas for better performance.
    (Copied from trend_detector_v1_gemini.py for testing purposes)
    """
    delta = series.diff().dropna()
    u = delta * 0
    d = delta * 0
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = np.abs(delta[delta < 0])
    rol_up = u.rolling(window=period).mean()
    rol_down = d.rolling(window=period).mean()
    rs = rol_up / rol_down
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_additional_features(df, fast_ma_period, slow_ma_period, rsi_period):
    """Calculate additional technical indicators.
    (Copied from trend_detector_v1_gemini.py for testing purposes)
    """
    df['MA_fast'] = df['close'].rolling(window=fast_ma_period).mean()
    df['MA_slow'] = df['close'].rolling(window=slow_ma_period).mean()
    df['RSI'] = calculate_rsi_pandas(df['close'], rsi_period) # Use pandas RSI
    df['Price_Change'] = df['close'].pct_change() # Percentage price change
    df.dropna(inplace=True) # Corrected: removed duplicate inplace
    return df


def prepare_training_data_test(df):
    """Prepare data for testing signal distribution.
    (Simplified signal logic, copied from trend_detector_v1_gemini.py)
    """
    df['Signal'] = 0  # 0: NEUTRAL
    # Simplified signal logic: Use only MA crossover for training data generation
    df.loc[df['MA_fast'] > df['MA_slow'], 'Signal'] = 1  # BUY if fast MA is above slow MA
    df.loc[df['MA_fast'] < df['MA_slow'], 'Signal'] = -1 # SELL if fast MA is below slow MA
    df.dropna(inplace=True) # Corrected: removed duplicate inplace
    return df['Signal'] # Only return the Signal column for analysis


def main():
    print(f"Fetching historical data for {SYMBOL}...")
    historical_df = fetch_binance_data(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=5000) # Fetch data

    if historical_df is None or historical_df.empty:
        print("Failed to fetch historical data.")
        return

    print("Calculating features...")
    feature_df = calculate_additional_features(historical_df.copy(), FAST_MA_PERIOD, SLOW_MA_PERIOD, RSI_PERIOD)

    print("Generating signals...")
    signal_series = prepare_training_data_test(feature_df.copy())

    print("\n--- Signal Distribution ---")
    print(signal_series.value_counts())

    print("\n--- First 10 rows with MAs and Signals ---")
    print(feature_df.head(10)) # Show first 10 rows of the feature DataFrame


if __name__ == "__main__":
    main() 