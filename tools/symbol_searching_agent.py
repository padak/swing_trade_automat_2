import os
from binance.client import Client
import pandas as pd
import pandas_ta as ta  # for technical indicators

# Binance API keys - make sure to set environment variables or replace with your keys
api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')
client = Client(api_key, api_secret)

def get_all_spot_symbols():
    """
    Fetches all SPOT trading symbols from Binance.
    Returns:
        list: A list of trading symbols (strings).
    """
    exchange_info = client.get_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols'] if s['quoteAsset'] in ['USDT', 'USDC'] and s['status'] == 'TRADING' and s['isSpotTradingAllowed']]
    return symbols

def fetch_historical_data(symbol, interval='1h', limit=100):
    """
    Fetches historical candlestick data for a given symbol from Binance.
    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str): Candlestick interval (e.g., '1h', '4h', '1d').
        limit (int): Number of historical data points to fetch.
    Returns:
        pd.DataFrame: DataFrame containing historical data, or None if error.
    """
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit + 1) # Fetch one extra to calculate returns properly
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'])
        df = df[['timestamp', 'close']] # Keep only timestamp and close price
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR) for a DataFrame with 'high', 'low', 'close' columns.
    Args:
        df (pd.DataFrame): DataFrame with historical price data.
        period (int): ATR period.
    Returns:
        pd.Series: ATR values.
    """
    if df is None or len(df) < period:
        return None
    atr_indicator = ta.volatility.AverageTrueRange(high=pd.to_numeric(df['high']), low=pd.to_numeric(df['low']), close=pd.to_numeric(df['close']), window=period)
    atr = atr_indicator.average_true_range()
    return atr.iloc[-1] # Return only the latest ATR value


def calculate_volatility_std_dev(df):
    """
    Calculates volatility based on the standard deviation of percentage price changes.
    Args:
        df (pd.DataFrame): DataFrame with historical price data ('timestamp', 'close').
    Returns:
        float: Standard deviation of percentage price changes, or None if insufficient data.
    """
    if df is None or len(df) < 2:
        return None
    df['price_change'] = df['close'].pct_change() * 100 # Calculate percentage change
    volatility = df['price_change'].std()
    return volatility

def analyze_symbols_volatility(symbols, interval='1h', limit=100, min_volume_usdt=100000):
    """
    Analyzes the volatility of a list of symbols.
    Args:
        symbols (list): List of trading symbols.
        interval (str): Candlestick interval for historical data.
        limit (int): Number of historical data points to fetch.
        min_volume_usdt: Minimum 24h volume in USDT to consider the symbol
    Returns:
        pd.DataFrame: DataFrame with symbol, volatility score and volume, sorted by volatility.
    """
    symbol_volatility = []
    
    # Get 24h ticker for all symbols at once (more efficient)
    tickers = client.get_ticker()
    volume_dict = {t['symbol']: float(t['quoteVolume']) for t in tickers}
    
    filtered_symbols = []
    for symbol in symbols:
        volume = volume_dict.get(symbol, 0)
        if volume >= min_volume_usdt:
            filtered_symbols.append(symbol)
        else:
            print(f"Skipping {symbol} due to low volume: ${volume:,.2f}")
            
    print(f"\nAnalyzing {len(filtered_symbols)} symbols with volume >= ${min_volume_usdt:,.2f}")
    
    for symbol in filtered_symbols:
        try:
            print(f"Fetching data and calculating volatility for {symbol}...")
            historical_data = fetch_historical_data(symbol, interval, limit)
            if historical_data is not None:
                volatility_score = calculate_volatility_std_dev(historical_data)
                if volatility_score is not None:
                    symbol_volatility.append({
                        'symbol': symbol,
                        'volatility_score': volatility_score,
                        'volume_24h': volume_dict[symbol]
                    })
                else:
                    print(f"Could not calculate volatility score for {symbol}.")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    volatility_df = pd.DataFrame(symbol_volatility)
    if not volatility_df.empty:
        volatility_df = volatility_df.sort_values(by='volatility_score', ascending=False).reset_index(drop=True)
        # Format volume as billions/millions for readability
        volatility_df['volume_24h_formatted'] = volatility_df['volume_24h'].apply(
            lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.2f}M"
        )
    return volatility_df

if __name__ == "__main__":
    print("Fetching all SPOT trading symbols...")
    spot_symbols = get_all_spot_symbols()
    print(f"Found {len(spot_symbols)} SPOT symbols.")

    print("\nAnalyzing volatility for all valid symbols...")
    volatile_symbols_df = analyze_symbols_volatility(
        symbols=spot_symbols,  # Analyze all symbols
        interval='1h',
        limit=100,
        min_volume_usdt=1000000  # Filter for symbols with >$1M 24h volume
    )

    if not volatile_symbols_df.empty:
        print("\n--- Top 30 Most Volatile Symbols ---")
        print(volatile_symbols_df[['symbol', 'volatility_score', 'volume_24h_formatted']].head(30))
    else:
        print("\nNo volatility data could be calculated for any symbol.")
