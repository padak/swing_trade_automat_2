import pandas as pd
import matplotlib.pyplot as plt

def analyze_trading_log(log_filepath):
    """
    Analyzes the trading log CSV file to identify performance and suggest improvements.
    Args:
        log_filepath (str): Path to the trading log CSV file.
    """
    try:
        df = pd.read_csv(log_filepath)
    except FileNotFoundError:
        print(f"Error: Trading log file not found at {log_filepath}")
        return

    # Convert 'Timestamp' to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # --- Basic Analysis ---
    print("--- Basic Trading Log Analysis ---")
    buy_orders = df[df['Trade_Action'] == 'BUY']
    sell_orders = df[df['Trade_Action'] == 'SELL']
    neutral_trades = df[df['Trade_Action'] == 'NEUTRAL']

    print(f"Total Log Entries: {len(df)}")
    print(f"Number of BUY orders: {len(buy_orders)}")
    print(f"Number of SELL orders: {len(sell_orders)}")
    print(f"Number of NEUTRAL iterations: {len(neutral_trades)}")

    if not df['P/L_Percent'].isnull().all():  # Check if P/L_Percent column is not all NaN
        try:
            final_portfolio_value = float(df['Portfolio_Value'].iloc[-1])
            initial_portfolio_value = float(df['Portfolio_Value'].iloc[0])
            total_pl_percent = float(df['P/L_Percent'].iloc[-1])
            print(f"Initial Portfolio Value: ${initial_portfolio_value:.2f}")
            print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
            print(f"Total P/L Percent: {total_pl_percent:.2f}%")
        except (ValueError, TypeError):
            print("Could not convert portfolio values to numbers - some values may be invalid")
    else:
        print("P/L_Percent data is not available for full performance analysis.")

    # --- Trend Following Analysis (Simple Example - can be expanded) ---
    print("\n--- Trend Following Opportunity Analysis (Simple) ---")
    uptrend_periods = df[df['Trend'] == 'UPTREND']
    neutral_signal_uptrend = uptrend_periods[uptrend_periods['Signal'] == 'NEUTRAL']

    if not neutral_signal_uptrend.empty:
        first_missed_uptrend_time = neutral_signal_uptrend.index.min()
        last_missed_uptrend_time = neutral_signal_uptrend.index.max()
        print(f"Periods with UPTREND but NEUTRAL signal found.")
        print(f"First occurrence: {first_missed_uptrend_time}")
        print(f"Last occurrence: {last_missed_uptrend_time}")
        print(f"Number of UPTREND & NEUTRAL signal entries: {len(neutral_signal_uptrend)}")

    # Example: Check price change during a missed uptrend period (can be refined)
    if not neutral_signal_uptrend.empty:
        try:
            start_price_missed_uptrend = float(df.loc[first_missed_uptrend_time]['Price'])
            end_price_missed_uptrend = float(df.loc[last_missed_uptrend_time]['Price'])
            price_change_missed_uptrend_percent = ((end_price_missed_uptrend - start_price_missed_uptrend) / start_price_missed_uptrend) * 100
            print(f"\nPrice change during first missed UPTREND period: {price_change_missed_uptrend_percent:.2f}%")
        except (ValueError, TypeError):
            print("\nCould not calculate price change - some price values may be invalid")
    else:
        print("No periods found with UPTREND and NEUTRAL signal.")

    # --- Visualization (Optional - uncomment to generate plots) ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['Portfolio_Value'], label='Portfolio Value')
    # plt.title('Portfolio Value Over Time')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Portfolio Value ($)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['Price'], label='Price', color='blue')
    # plt.plot(df.index, df['MA_Fast'], label='MA_Fast', color='orange')
    # plt.plot(df.index, df['MA_Slow'], label='MA_Slow', color='green')
    # plt.scatter(buy_orders.index, buy_orders['Price'], color='green', marker='^', label='BUY Orders')
    # plt.scatter(sell_orders.index, sell_orders['Price'], color='red', marker='v', label='SELL Orders')
    # plt.title('Price, Moving Averages, and Trade Orders')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    log_file = 'src/trading_log.csv'  # Replace with the actual path to your log file if needed
    analyze_trading_log(log_file)