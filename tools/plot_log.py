import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Add constants for directories
LOGS_DIR = "logs"
TRADING_LOG_PATH = os.path.join(LOGS_DIR, "trading_log.csv")

def plot_trading_log(log_filepath=TRADING_LOG_PATH):
    """
    Reads a trading log CSV file, plots portfolio value, price, and trade signals over time.

    Args:
        log_filepath (str): Path to the trading log CSV file.
    """
    try:
        # Check if logs directory exists
        if not os.path.exists(LOGS_DIR):
            print(f"Error: Logs directory not found at: {LOGS_DIR}")
            print("Please run the trend detector first to generate trading data.")
            return

        # Read and validate the data
        df = pd.read_csv(log_filepath)
        if df.empty:
            print(f"Error: Log file '{log_filepath}' is empty.")
            return

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Create the plots directory if it doesn't exist
        plots_dir = os.path.join(LOGS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create figure with three subplots
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)
        fig.suptitle('Trading Simulation Performance Over Time', fontsize=16)

        # Plot Portfolio Value
        axes[0].plot(df.index, df['Portfolio_Value'], label='Portfolio Value (USDC)', color='blue')
        axes[0].set_ylabel('Portfolio Value (USDC)')
        axes[0].set_title('Portfolio Value')
        axes[0].grid(True)
        axes[0].legend(loc='upper left')

        # Plot Price with Trends
        axes[1].plot(df.index, df['Price'], label='Price', color='purple')
        axes[1].set_ylabel('Price')
        axes[1].set_title('Market Price and Trends')
        axes[1].grid(True)

        # Color background based on trend
        for i in range(len(df)-1):
            trend = df['Trend'].iloc[i]
            color = 'lightgreen' if trend == 'UPTREND' else 'lightcoral' if trend == 'DOWNTREND' else 'lightgray'
            axes[1].axvspan(df.index[i], df.index[i+1], alpha=0.3, color=color)

        # Plot Trade Signals
        axes[2].plot(df.index, df['Price'], label='Price', color='purple')
        axes[2].set_ylabel('Price')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Trade Signals')
        axes[2].grid(True)

        # Add Buy/Sell signals with annotations
        buy_signals = df[df['Trade_Action'] == 'BUY']
        sell_signals = df[df['Trade_Action'] == 'SELL']

        # Plot signals on both price charts
        for ax in [axes[1], axes[2]]:
            # Plot BUY signals
            ax.scatter(buy_signals.index, buy_signals['Price'], 
                      color='green', marker='^', s=100, label='BUY Signal')
            
            # Plot SELL signals
            ax.scatter(sell_signals.index, sell_signals['Price'], 
                      color='red', marker='v', s=100, label='SELL Signal')

            # Add annotations for signals
            for idx, row in buy_signals.iterrows():
                ax.annotate(f'BUY\n${float(row["Price"]):.2f}', 
                           (idx, row['Price']), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           color='green',
                           fontweight='bold')

            for idx, row in sell_signals.iterrows():
                ax.annotate(f'SELL\n${float(row["Price"]):.2f}', 
                           (idx, row['Price']), 
                           xytext=(10, -20), 
                           textcoords='offset points',
                           color='red',
                           fontweight='bold')

            ax.legend()

        # Format x-axis to show dates nicely
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plot_path = os.path.join(plots_dir, 'trading_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plot_path}")
        
        # Display plot
        plt.show()

        # Print statistics (with fixed formatting)
        print("\n=== Trading Statistics ===")
        print(f"Total Trades: {len(buy_signals) + len(sell_signals)}")
        print(f"Buy Signals: {len(buy_signals)}")
        print(f"Sell Signals: {len(sell_signals)}")
        
        try:
            # Clean and convert portfolio values
            df['Portfolio_Value'] = df['Portfolio_Value'].replace(['-', ''], '0')  # Replace '-' and empty values with '0'
            initial_value = float(df['Portfolio_Value'].iloc[0])
            final_value = float(df['Portfolio_Value'].iloc[-1])
            
            print(f"Initial Portfolio Value: ${initial_value:.2f}")
            print(f"Final Portfolio Value: ${final_value:.2f}")
            
            if initial_value > 0:  # Only calculate return if initial value is positive
                print(f"Total Return: {((final_value / initial_value) - 1) * 100:.2f}%")
            else:
                print("Total Return: N/A (insufficient data)")
        except Exception as e:
            print("Could not calculate portfolio statistics due to invalid data")
            print(f"Error details: {e}")

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_filepath}'.")
        print("Please run the trend detector first to generate trading data.")
    except pd.errors.EmptyDataError:
        print(f"Error: No data in log file '{log_filepath}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_trading_log() 