import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_trading_log(log_filepath='trading_log.csv'):
    """
    Reads a trading log CSV file, plots portfolio value, price, and trade signals over time.

    Args:
        log_filepath (str): Path to the trading log CSV file.
    """
    try:
        df = pd.read_csv(log_filepath)
        if df.empty:
            print(f"Error: Log file '{log_filepath}' is empty.")
            return

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True) # Two subplots, shared x-axis
        fig.suptitle('Trading Simulation Performance Over Time', fontsize=16)

        # Plot Portfolio Value
        axes[0].plot(df.index, df['Portfolio_Value'], label='Portfolio Value (USDC)', color='blue')
        axes[0].set_ylabel('Portfolio Value (USDC)')
        axes[0].set_title('Portfolio Value and Price')
        axes[0].grid(True)
        axes[0].legend(loc='upper left')

        # Plot Price and Trade Signals
        axes[1].plot(df.index, df['Price'], label='Price', color='purple')
        axes[1].set_ylabel('Price')
        axes[1].set_xlabel('Time')
        axes[1].grid(True)
        axes[1].legend(loc='upper left')

        # --- Highlight BUY and SELL signals on the Price chart ---
        buy_signals = df[df['Trade_Action'] == 'BUY']
        sell_signals = df[df['Trade_Action'] == 'SELL']

        axes[1].scatter(buy_signals.index, buy_signals['Price'], color='green', marker='^', s=50, label='BUY Signal') # Green triangles for BUY
        axes[1].scatter(sell_signals.index, sell_signals['Price'], color='red', marker='v', s=50, label='SELL Signal') # Red inverted triangles for SELL
        axes[1].legend() # Update legend to include signals

        # Format x-axis to show dates nicely
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # Date and time format
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap and title cut-off
        plt.show()

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_filepath}'.")
    except pd.errors.EmptyDataError:
        print(f"Error: No data in log file '{log_filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_trading_log() 