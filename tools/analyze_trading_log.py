import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
import os

# Add constants for directories
LOGS_DIR = "logs"
TRADING_LOG_PATH = os.path.join(LOGS_DIR, "trading_log.csv")

def calculate_optimal_rsi_levels(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate optimal RSI levels using vectorized operations.
    """
    # Vectorized approach for RSI optimization
    buy_levels = np.arange(20, 41)
    sell_levels = np.arange(60, 81)
    
    # Create grid of all possible combinations
    buy_grid, sell_grid = np.meshgrid(buy_levels, sell_levels)
    
    # Initialize profit matrix
    profit_matrix = np.zeros_like(buy_grid, dtype=np.float64)
    
    # Vectorized price changes
    price_changes = df['Price'].diff().values
    rsi_values = df['RSI'].values
    
    # Pre-calculate position changes
    buy_signals = (rsi_values[:-1] <= buy_grid[:, :, None]) & (price_changes[1:] > 0)
    sell_signals = (rsi_values[:-1] >= sell_grid[:, :, None]) & (price_changes[1:] < 0)
    
    # Matrix operations for profit calculation
    for i in range(buy_grid.shape[0]):
        for j in range(buy_grid.shape[1]):
            positions = np.zeros(len(df))
            entries = np.where(buy_signals[i, j])[0]
            exits = np.where(sell_signals[i, j])[0]
            
            # Vectorized profit calculation
            if len(entries) > 0 and len(exits) > 0:
                valid_trades = exits > entries[0]
                if np.any(valid_trades):
                    first_exit = exits[valid_trades][0]
                    profit = (df['Price'].iloc[first_exit] - df['Price'].iloc[entries[0]]) / df['Price'].iloc[entries[0]] * 100
                    profit_matrix[i, j] = profit
                    
    # Find optimal levels
    max_idx = np.argmax(profit_matrix)
    optimal_buy = buy_grid.flat[max_idx]
    optimal_sell = sell_grid.flat[max_idx]
    
    return optimal_buy, optimal_sell

def analyze_ma_effectiveness(df: pd.DataFrame) -> Dict:
    """
    Analyze the effectiveness of Moving Average crossovers.
    """
    df['MA_Cross'] = np.where(df['MA_Fast'] > df['MA_Slow'], 1, -1)
    df['MA_Cross_Change'] = df['MA_Cross'].diff()
    
    results = {
        'total_crosses': 0,
        'profitable_crosses': 0,
        'avg_profit_per_cross': 0,
        'best_cross_profit': 0,
        'worst_cross_loss': 0
    }
    
    # Analyze each MA crossover
    cross_points = df[df['MA_Cross_Change'] != 0].index
    for i in range(len(cross_points)-1):
        start_idx = cross_points[i]
        end_idx = cross_points[i+1]
        
        if df.loc[start_idx, 'MA_Cross_Change'] > 0:  # Bullish crossover
            price_change = (df.loc[end_idx, 'Price'] - df.loc[start_idx, 'Price']) / df.loc[start_idx, 'Price'] * 100
        else:  # Bearish crossover
            price_change = (df.loc[start_idx, 'Price'] - df.loc[end_idx, 'Price']) / df.loc[start_idx, 'Price'] * 100
            
        results['total_crosses'] += 1
        if price_change > 0:
            results['profitable_crosses'] += 1
        results['avg_profit_per_cross'] += price_change
        results['best_cross_profit'] = max(results['best_cross_profit'], price_change)
        results['worst_cross_loss'] = min(results['worst_cross_loss'], price_change)
    
    if results['total_crosses'] > 0:
        results['avg_profit_per_cross'] /= results['total_crosses']
        
    return results

def analyze_missed_opportunities(df: pd.DataFrame) -> Dict:
    """
    Analyze missed trading opportunities and false signals.
    """
    results = {
        'missed_uptrends': 0,
        'missed_downtrends': 0,
        'false_buy_signals': 0,
        'false_sell_signals': 0,
        'profit_missed_uptrends': 0,
        'profit_missed_downtrends': 0
    }
    
    # Analyze missed opportunities
    for i in range(1, len(df)-1):
        # Missed uptrends
        if df['Trend'].iloc[i] == 'UPTREND' and df['Signal'].iloc[i] == 'NEUTRAL':
            if df['Price'].iloc[i+1] > df['Price'].iloc[i]:
                results['missed_uptrends'] += 1
                results['profit_missed_uptrends'] += (df['Price'].iloc[i+1] - df['Price'].iloc[i]) / df['Price'].iloc[i] * 100
                
        # Missed downtrends
        if df['Trend'].iloc[i] == 'DOWNTREND' and df['Signal'].iloc[i] == 'NEUTRAL':
            if df['Price'].iloc[i+1] < df['Price'].iloc[i]:
                results['missed_downtrends'] += 1
                results['profit_missed_downtrends'] += (df['Price'].iloc[i] - df['Price'].iloc[i+1]) / df['Price'].iloc[i] * 100
                
        # False signals
        if df['Signal'].iloc[i] == 'BUY' and df['Price'].iloc[i+1] < df['Price'].iloc[i]:
            results['false_buy_signals'] += 1
        if df['Signal'].iloc[i] == 'SELL' and df['Price'].iloc[i+1] > df['Price'].iloc[i]:
            results['false_sell_signals'] += 1
    
    return results

def analyze_macd_effectiveness(df: pd.DataFrame) -> Dict:
    """
    Analyze MACD signal effectiveness.
    """
    results = {
        'macd_crosses': 0,
        'successful_macd_signals': 0,
        'avg_profit_per_macd_signal': 0,
        'best_macd_profit': 0,
        'worst_macd_loss': 0
    }
    
    df['MACD_Cross'] = np.where(df['MACD_Line'] > df['MACD_Signal'], 1, -1)
    df['MACD_Cross_Change'] = df['MACD_Cross'].diff()
    
    # Analyze each MACD crossover
    cross_points = df[df['MACD_Cross_Change'] != 0].index
    for i in range(len(cross_points)-1):
        start_idx = cross_points[i]
        end_idx = cross_points[i+1]
        
        if df.loc[start_idx, 'MACD_Cross_Change'] > 0:  # Bullish crossover
            price_change = (df.loc[end_idx, 'Price'] - df.loc[start_idx, 'Price']) / df.loc[start_idx, 'Price'] * 100
        else:  # Bearish crossover
            price_change = (df.loc[start_idx, 'Price'] - df.loc[end_idx, 'Price']) / df.loc[start_idx, 'Price'] * 100
            
        results['macd_crosses'] += 1
        if price_change > 0:
            results['successful_macd_signals'] += 1
        results['avg_profit_per_macd_signal'] += price_change
        results['best_macd_profit'] = max(results['best_macd_profit'], price_change)
        results['worst_macd_loss'] = min(results['worst_macd_loss'], price_change)
    
    if results['macd_crosses'] > 0:
        results['avg_profit_per_macd_signal'] /= results['macd_crosses']
        
    return results

def analyze_trend_detection(df: pd.DataFrame) -> Dict:
    """
    Analyze the accuracy of trend detection against actual price movements.
    """
    results = {
        'uptrend_correct': 0,
        'downtrend_correct': 0,
        'neutral_correct': 0,
        'total_uptrends': 0,
        'total_downtrends': 0,
        'total_neutral': 0,
        'false_positive_uptrend': 0,
        'false_positive_downtrend': 0,
        'uptrend_accuracy': 0.0,
        'downtrend_accuracy': 0.0,
        'neutral_accuracy': 0.0
    }
    
    # Calculate actual price trends using 3-period momentum
    df['Price_Change'] = df['Price'].pct_change(3)
    df['Actual_Trend'] = np.select(
        [
            df['Price_Change'] > 0.01,
            df['Price_Change'] < -0.01
        ],
        ['UPTREND', 'DOWNTREND'],
        default='NEUTRAL'
    )
    
    for i in range(3, len(df)):
        # Model's trend detection
        model_trend = df['Trend'].iloc[i]
        actual_trend = df['Actual_Trend'].iloc[i]
        
        # Count occurrences
        if actual_trend == 'UPTREND':
            results['total_uptrends'] += 1
            if model_trend == 'UPTREND':
                results['uptrend_correct'] += 1
        elif actual_trend == 'DOWNTREND':
            results['total_downtrends'] += 1
            if model_trend == 'DOWNTREND':
                results['downtrend_correct'] += 1
        else:
            results['total_neutral'] += 1
            if model_trend == 'NEUTRAL':
                results['neutral_correct'] += 1
                
        # Count false positives
        if model_trend == 'UPTREND' and actual_trend != 'UPTREND':
            results['false_positive_uptrend'] += 1
        if model_trend == 'DOWNTREND' and actual_trend != 'DOWNTREND':
            results['false_positive_downtrend'] += 1
    
    # Calculate accuracies with zero division protection
    if results['total_uptrends'] > 0:
        results['uptrend_accuracy'] = results['uptrend_correct'] / results['total_uptrends']
    if results['total_downtrends'] > 0:
        results['downtrend_accuracy'] = results['downtrend_correct'] / results['total_downtrends']
    if results['total_neutral'] > 0:
        results['neutral_accuracy'] = results['neutral_correct'] / results['total_neutral']
            
    return results

def find_leading_indicators(df: pd.DataFrame) -> Dict:
    """
    Identify which indicators lead price movements most effectively.
    """
    leads = {
        'RSI_lead': 0,
        'MACD_lead': 0,
        'MA_lead': 0,
        'total_changes': 0,
        'RSI_accuracy': 0.0,
        'MACD_accuracy': 0.0,
        'MA_accuracy': 0.0
    }
    
    # Calculate future price changes
    df['Future_3_Change'] = df['Price'].pct_change(3).shift(-3)
    
    for i in range(len(df)-4):
        # Look for indicator signals before price moves
        if df['RSI'].iloc[i] < 30 and df['Future_3_Change'].iloc[i] > 0.02:
            leads['RSI_lead'] += 1
        if df['MACD_Line'].iloc[i] > df['MACD_Signal'].iloc[i] and df['Future_3_Change'].iloc[i] > 0.02:
            leads['MACD_lead'] += 1
        if df['MA_Fast'].iloc[i] > df['MA_Slow'].iloc[i] and df['Future_3_Change'].iloc[i] > 0.02:
            leads['MA_lead'] += 1
            
        if abs(df['Future_3_Change'].iloc[i]) > 0.02:
            leads['total_changes'] += 1
    
    # Calculate accuracies with zero division protection
    if leads['total_changes'] > 0:
        leads['RSI_accuracy'] = leads['RSI_lead'] / leads['total_changes']
        leads['MACD_accuracy'] = leads['MACD_lead'] / leads['total_changes']
        leads['MA_accuracy'] = leads['MA_lead'] / leads['total_changes']
            
    return leads

def calculate_indicator_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix between indicators and future price changes.
    """
    # Create future price change features
    df['Next_1_Change'] = df['Price'].pct_change().shift(-1)
    df['Next_3_Change'] = df['Price'].pct_change(3).shift(-3)
    
    # Select relevant columns for correlation
    corr_columns = [
        'RSI', 'MACD_Line', 'MA_Fast', 'MA_Slow',
        'Next_1_Change', 'Next_3_Change'
    ]
    
    return df[corr_columns].corr()

def calculate_rsi(prices, period=14):
    """
    Calculate RSI for a price series
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def suggest_trend_detector_adjustments(trend_results: dict) -> dict:
    """
    Based on uptrend/downtrend detection accuracy, propose changes
    to RSI thresholds and moving average periods to improve performance.
    Example heuristic approach for demonstration.
    """
    recommended = {
        "new_rsi_overbought": None,
        "new_rsi_oversold": None,
        "new_fast_ma_period": None,
        "new_slow_ma_period": None
    }

    # Heuristic rules to adjust RSI
    # e.g. if we have poor uptrend detection, we want RSI oversold higher (less conservative)
    # if we have poor downtrend detection, we want RSI overbought lower.
    uptrend_acc = trend_results.get('uptrend_accuracy', 0)
    downtrend_acc = trend_results.get('downtrend_accuracy', 0)

    # Start with placeholders (assumes default 70/30 for RSI)
    rsi_overbought = 70
    rsi_oversold = 30
    fast_ma = 10
    slow_ma = 30

    # If uptrend detection is under 70%, make RSI oversold more lenient (increase)
    if uptrend_acc < 0.70:
        rsi_oversold += 5  # more lenient buy condition

    # If downtrend detection is under 70%, make RSI overbought more aggressive (decrease)
    if downtrend_acc < 0.70:
        rsi_overbought -= 5  # more aggressive sell condition

    recommended["new_rsi_overbought"] = rsi_overbought
    recommended["new_rsi_oversold"] = rsi_oversold

    # Optionally, tweak fast/slow MA if both uptrend and downtrend performance is weak
    if uptrend_acc < 0.60 and downtrend_acc < 0.60:
        fast_ma = 8  # example minor shift
        slow_ma = 28

    recommended["new_fast_ma_period"] = fast_ma
    recommended["new_slow_ma_period"] = slow_ma

    return recommended

def analyze_trading_log(log_filepath):
    """
    Analyze trading log with proper data type handling
    """
    # Define column types for numeric data
    dtype_dict = {
        'Price': 'float64',
        'USDC_Balance': 'float64',
        'TRUMP_Balance': 'float64',
        'Portfolio_Value': 'float64',
        'P/L_Percent': 'float64',
        'Trade_Quantity': 'float64',
        'Trade_Price': 'float64',
        'RSI_Overbought': 'float64',
        'RSI_Oversold': 'float64',
        'MA_Fast': 'float64',
        'MA_Slow': 'float64',
        'MACD_Line': 'float64',
        'MACD_Signal': 'float64'
    }
    
    # Load data with proper types and handle missing values
    df = pd.read_csv(log_filepath, 
                     dtype=dtype_dict,
                     na_values=['n/a', 'nan', '-'],  # Handle n/a values
                     parse_dates=['Timestamp'])  # Parse timestamp column
    
    # Remove rows that are parameter adjustments or reports
    df = df[~df['Signal'].isin(['PARAMS_ADJUST', 'REPORT'])]
    
    # Fill NaN values appropriately
    df = df.fillna({
        'Trade_Quantity': 0,
        'Trade_Price': df['Price'],
        'Win_Rate': 0,
        'Avg_Trade_Duration': 0
    })

    # Calculate RSI if not present (you might need to implement this)
    if 'RSI' not in df.columns:
        df['RSI'] = calculate_rsi(df['Price'])  # You'll need to implement this function

    print("\n=== Enhanced Trading Analysis Report ===")
    
    # Basic statistics
    print("\n--- Basic Trading Statistics ---")
    total_trades = len(df[df['Trade_Action'].isin(['BUY', 'SELL'])])
    buy_trades = len(df[df['Trade_Action'] == 'BUY'])
    sell_trades = len(df[df['Trade_Action'] == 'SELL'])
    
    print(f"Total Trades: {total_trades}")
    print(f"Buy Orders: {buy_trades}")
    print(f"Sell Orders: {sell_trades}")
    
    # Overall performance
    print(f"\nOverall Performance: {df['P/L_Percent'].iloc[-1]:.2f}%")

    # Add analysis execution
    print("\n--- RSI Analysis ---")
    try:
        optimal_buy, optimal_sell = calculate_optimal_rsi_levels(df)
        print(f"Optimal RSI Levels: Buy at {optimal_buy}, Sell at {optimal_sell}")
    except Exception as e:
        print("RSI analysis failed:", str(e))

    print("\n--- Moving Average Analysis ---")
    try:
        ma_results = analyze_ma_effectiveness(df)
        print(f"MA Cross Effectiveness: {ma_results['profitable_crosses']}/{ma_results['total_crosses']} profitable crosses")
        print(f"Average profit per MA cross: {ma_results['avg_profit_per_cross']:.2f}%")
    except Exception as e:
        print("MA analysis failed:", str(e))

    print("\n--- MACD Analysis ---")
    try:
        macd_results = analyze_macd_effectiveness(df)
        print(f"MACD Effectiveness: {macd_results['successful_macd_signals']}/{macd_results['macd_crosses']} successful signals")
        print(f"Average MACD signal profit: {macd_results['avg_profit_per_macd_signal']:.2f}%")
    except Exception as e:
        print("MACD analysis failed:", str(e))

    print("\n--- Trend Detection Accuracy ---")
    try:
        trend_results = analyze_trend_detection(df)
        print(f"Uptrend Detection Accuracy: {trend_results['uptrend_accuracy']:.1%}")
        print(f"Downtrend Detection Accuracy: {trend_results['downtrend_accuracy']:.1%}")
        print(f"Total Uptrends Detected: {trend_results['total_uptrends']}")
        print(f"Total Downtrends Detected: {trend_results['total_downtrends']}")
        print(f"False Positive Uptrends: {trend_results['false_positive_uptrend']}")
        print(f"False Positive Downtrends: {trend_results['false_positive_downtrend']}")

        # --------------------------------------------------------
        # NEW: Automatically suggest parameter adjustments
        suggested_params = suggest_trend_detector_adjustments(trend_results)
        print("\n--- Suggested Adjustments for trend_detector_v2_gemini.py ---")
        print("Adjust RSI_Overbought to:", suggested_params["new_rsi_overbought"])
        print("Adjust RSI_Oversold to:", suggested_params["new_rsi_oversold"])
        print("Adjust FAST_MA_PERIOD to:", suggested_params["new_fast_ma_period"])
        print("Adjust SLOW_MA_PERIOD to:", suggested_params["new_slow_ma_period"])
        # --------------------------------------------------------
        
    except Exception as e:
        print("Trend analysis failed:", str(e))

    print("\n--- Leading Indicators Analysis ---")
    try:
        lead_results = find_leading_indicators(df)
        print(f"RSI Leading Accuracy: {lead_results['RSI_accuracy']:.1%}")
        print(f"MACD Leading Accuracy: {lead_results['MACD_accuracy']:.1%}")
        print(f"MA Leading Accuracy: {lead_results['MA_accuracy']:.1%}")
        print(f"Total Price Changes Analyzed: {lead_results['total_changes']}")
        print(f"RSI Successful Predictions: {lead_results['RSI_lead']}")
        print(f"MACD Successful Predictions: {lead_results['MACD_lead']}")
        print(f"MA Successful Predictions: {lead_results['MA_lead']}")
    except Exception as e:
        print("Leading indicator analysis failed:", str(e))

    # Rest of your analysis code...

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Check if trading log exists
    if not os.path.exists(TRADING_LOG_PATH):
        print(f"Error: Trading log file not found at {TRADING_LOG_PATH}")
        print("Please run the trend detector first to generate trading data.")
        exit(1)
        
    print(f"Analyzing trading log from: {TRADING_LOG_PATH}")
    analyze_trading_log(TRADING_LOG_PATH)
    
    # Save the visualization in the logs directory
    plt.savefig(os.path.join(LOGS_DIR, 'trading_analysis.png'))
    print(f"\nVisualization saved as: {os.path.join(LOGS_DIR, 'trading_analysis.png')}")