import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

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
        'false_positive_downtrend': 0
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
            
    return results

def find_leading_indicators(df: pd.DataFrame) -> Dict:
    """
    Identify which indicators lead price movements most effectively.
    """
    leads = {
        'RSI_lead': 0,
        'MACD_lead': 0,
        'MA_lead': 0,
        'total_changes': 0
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

def analyze_trading_log(log_filepath):
    """
    Analyzes the trading log CSV file to identify performance and suggest improvements.
    Args:
        log_filepath (str): Path to the trading log CSV file.
    """
    try:
        # Load data with optimized types
        dtype_dict = {
            'RSI': 'float32',
            'MA_Fast': 'float32',
            'MA_Slow': 'float32',
            'MACD_Line': 'float32',
            'MACD_Signal': 'float32'
        }
        df = pd.read_csv(log_filepath, dtype=dtype_dict)
    except FileNotFoundError:
        print(f"Error: Trading log file not found at {log_filepath}")
        return

    # Convert 'Timestamp' to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # Calculate RSI (if not already in the dataframe)
    if 'RSI' not in df.columns:
        df['RSI'] = df['RSI_Overbought'].fillna(method='ffill')  # Using overbought level as proxy

    print("\n=== Enhanced Trading Analysis Report ===\n")

    # Basic statistics
    print("--- Basic Trading Statistics ---")
    buy_orders = df[df['Trade_Action'] == 'BUY']
    sell_orders = df[df['Trade_Action'] == 'SELL']
    print(f"Total Trades: {len(buy_orders) + len(sell_orders)}")
    print(f"Buy Orders: {len(buy_orders)}")
    print(f"Sell Orders: {len(sell_orders)}")
    
    # Performance metrics
    if not df['P/L_Percent'].isnull().all():
        total_pl_percent = float(df['P/L_Percent'].iloc[-1])
        print(f"\nOverall Performance: {total_pl_percent:.2f}%")
        
    # Optimal RSI Levels Analysis
    print("\n--- RSI Analysis ---")
    try:
        optimal_buy_rsi, optimal_sell_rsi = calculate_optimal_rsi_levels(df)
        print(f"Suggested Optimal RSI Levels:")
        print(f"Buy Level: {optimal_buy_rsi}")
        print(f"Sell Level: {optimal_sell_rsi}")
        print(f"Current RSI Levels: Buy={df['RSI_Oversold'].iloc[-1]}, Sell={df['RSI_Overbought'].iloc[-1]}")
    except Exception as e:
        print("Could not calculate optimal RSI levels:", str(e))

    # Moving Average Analysis
    print("\n--- Moving Average Analysis ---")
    try:
        ma_results = analyze_ma_effectiveness(df)
        print(f"MA Crossover Effectiveness:")
        print(f"Total Crossovers: {ma_results['total_crosses']}")
        print(f"Profitable Crossovers: {ma_results['profitable_crosses']}")
        print(f"Average Profit per Crossover: {ma_results['avg_profit_per_cross']:.2f}%")
        print(f"Best Crossover Profit: {ma_results['best_cross_profit']:.2f}%")
        print(f"Worst Crossover Loss: {ma_results['worst_cross_loss']:.2f}%")
    except Exception as e:
        print("Could not analyze MA effectiveness:", str(e))

    # Missed Opportunities Analysis
    print("\n--- Missed Opportunities Analysis ---")
    try:
        missed_opp = analyze_missed_opportunities(df)
        print(f"Missed Uptrends: {missed_opp['missed_uptrends']}")
        print(f"Missed Downtrends: {missed_opp['missed_downtrends']}")
        print(f"False Buy Signals: {missed_opp['false_buy_signals']}")
        print(f"False Sell Signals: {missed_opp['false_sell_signals']}")
        print(f"Potential Profit Missed (Uptrends): {missed_opp['profit_missed_uptrends']:.2f}%")
        print(f"Potential Profit Missed (Downtrends): {missed_opp['profit_missed_downtrends']:.2f}%")
    except Exception as e:
        print("Could not analyze missed opportunities:", str(e))

    # MACD Analysis
    print("\n--- MACD Analysis ---")
    try:
        macd_results = analyze_macd_effectiveness(df)
        print(f"MACD Signal Effectiveness:")
        print(f"Total MACD Crosses: {macd_results['macd_crosses']}")
        print(f"Successful Signals: {macd_results['successful_macd_signals']}")
        print(f"Average Profit per Signal: {macd_results['avg_profit_per_macd_signal']:.2f}%")
        print(f"Best MACD Profit: {macd_results['best_macd_profit']:.2f}%")
        print(f"Worst MACD Loss: {macd_results['worst_macd_loss']:.2f}%")
    except Exception as e:
        print("Could not analyze MACD effectiveness:", str(e))

    # Trend Detection Accuracy
    print("\n--- Trend Detection Accuracy ---")
    try:
        trend_results = analyze_trend_detection(df)
        print(f"Uptrend Detection Accuracy: {trend_results['uptrend_correct']/trend_results['total_uptrends']:.1%}")
        print(f"Downtrend Detection Accuracy: {trend_results['downtrend_correct']/trend_results['total_downtrends']:.1%}")
        print(f"False Positive Uptrends: {trend_results['false_positive_uptrend']}")
        print(f"False Positive Downtrends: {trend_results['false_positive_downtrend']}")
    except Exception as e:
        print("Trend analysis failed:", str(e))

    # Leading Indicators Analysis
    print("\n--- Leading Indicators Analysis ---")
    try:
        lead_results = find_leading_indicators(df)
        print(f"RSI Leading Accuracy: {lead_results['RSI_lead']/lead_results['total_changes']:.1%}")
        print(f"MACD Leading Accuracy: {lead_results['MACD_lead']/lead_results['total_changes']:.1%}")
        print(f"MA Crossover Leading Accuracy: {lead_results['MA_lead']/lead_results['total_changes']:.1%}")
    except Exception as e:
        print("Leading indicator analysis failed:", str(e))

    # Indicator Correlations
    print("\n--- Indicator Correlations ---")
    try:
        corr_matrix = calculate_indicator_correlations(df)
        print("Correlation with Future Price Changes:")
        print(corr_matrix[['Next_1_Change', 'Next_3_Change']].sort_values('Next_3_Change', ascending=False))
    except Exception as e:
        print("Correlation analysis failed:", str(e))

    # Model Tuning Recommendations
    print("\n=== Model Tuning Recommendations ===")
    recommendations = []
    
    # RSI Recommendations
    if 'optimal_buy_rsi' in locals() and 'optimal_sell_rsi' in locals():
        current_buy_rsi = df['RSI_Oversold'].iloc[-1]
        current_sell_rsi = df['RSI_Overbought'].iloc[-1]
        if abs(optimal_buy_rsi - current_buy_rsi) > 5:
            recommendations.append(f"Consider adjusting RSI buy level to {optimal_buy_rsi}")
        if abs(optimal_sell_rsi - current_sell_rsi) > 5:
            recommendations.append(f"Consider adjusting RSI sell level to {optimal_sell_rsi}")

    # MA Recommendations
    if 'ma_results' in locals():
        if ma_results['profitable_crosses'] / ma_results['total_crosses'] < 0.5:
            recommendations.append("Consider increasing MA crossover confirmation period")
        if ma_results['worst_cross_loss'] < -2:
            recommendations.append("Consider adding stop-loss at -2% for MA crossover trades")

    # MACD Recommendations
    if 'macd_results' in locals():
        if macd_results['successful_macd_signals'] / macd_results['macd_crosses'] < 0.5:
            recommendations.append("Consider using MACD only for trend confirmation, not primary signals")

    # Missed Opportunities Recommendations
    if 'missed_opp' in locals():
        if missed_opp['missed_uptrends'] > missed_opp['false_buy_signals']:
            recommendations.append("Consider relaxing buy conditions to catch more uptrends")
        if missed_opp['missed_downtrends'] > missed_opp['false_sell_signals']:
            recommendations.append("Consider relaxing sell conditions to catch more downtrends")

    # Trend Recommendations
    if 'trend_results' in locals():
        if trend_results['false_positive_uptrend'] > trend_results['uptrend_correct']:
            recommendations.append("Consider increasing confirmation period for uptrend detection")
        if trend_results['false_positive_downtrend'] > trend_results['downtrend_correct']:
            recommendations.append("Consider adding volume confirmation for downtrend detection")

    # Leading Indicator Recommendations
    if 'lead_results' in locals():
        best_lead = max(['RSI_lead', 'MACD_lead', 'MA_lead'], 
                       key=lambda x: lead_results[x]/lead_results['total_changes'])
        recommendations.append(f"Prioritize {best_lead.replace('_lead', '')} signals for earlier entries")

    print("\n".join(recommendations))

    # Optional: Generate visualizations
    try:
        # Downsample data for visualization
        plot_df = df.iloc[::10]  # Use every 10th data point
        
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(plot_df.index, plot_df['Price'], label='Price')
        plt.plot(plot_df.index, plot_df['MA_Fast'], label='Fast MA')
        plt.plot(plot_df.index, plot_df['MA_Slow'], label='Slow MA')
        plt.title('Price and Moving Averages')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(plot_df.index, plot_df['RSI'], label='RSI')
        plt.axhline(y=plot_df['RSI_Oversold'].iloc[-1], color='g', linestyle='--', label='Oversold')
        plt.axhline(y=plot_df['RSI_Overbought'].iloc[-1], color='r', linestyle='--', label='Overbought')
        plt.title('RSI Indicator')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(plot_df.index, plot_df['MACD_Line'], label='MACD')
        plt.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal')
        plt.title('MACD')
        plt.legend()

        plt.tight_layout()
        plt.savefig('trading_analysis.png')
        print("\nVisualization saved as 'trading_analysis.png'")
    except Exception as e:
        print("Could not generate visualizations:", str(e))

    # Add visualization for trend detection
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df['Price'], label='Price')
        plt.scatter(plot_df[plot_df['Actual_Trend'] == 'UPTREND'].index, 
                   plot_df[plot_df['Actual_Trend'] == 'UPTREND']['Price'],
                   color='g', alpha=0.3, label='Actual Uptrend')
        plt.scatter(plot_df[plot_df['Trend'] == 'UPTREND'].index,
                   plot_df[plot_df['Trend'] == 'UPTREND']['Price'],
                   marker='x', color='lime', label='Detected Uptrend')
        plt.title('Trend Detection Accuracy')
        plt.legend()
        plt.savefig('trend_detection.png')
        print("\nTrend visualization saved as 'trend_detection.png'")
    except Exception as e:
        print("Could not generate trend visualization:", str(e))

if __name__ == "__main__":
    log_file = 'src/trading_log.csv'  # Replace with the actual path to your log file if needed
    analyze_trading_log(log_file)