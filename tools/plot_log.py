import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Update the constant to use trading_log_v3.csv
LOGS_DIR = "logs"
TRADING_LOG_PATH = os.path.join(LOGS_DIR, "trading_log_v3.csv")

def plot_trading_log(log_filepath=TRADING_LOG_PATH):
    """
    Reads a trading log CSV file, plots market price, areas with uptrend and downtrend,
    displays generated signals (buy/sell) and overlays a simulation of the potenial portfolio value 
    if optimal trades were made at trend reversals.
    
    The simulation assumes an initial portfolio of 500 USDC cash and an additional 500 USDC value in TRUMP.
    Optimal trades are executed when the trend reverses: buy when trend changes from DOWNTREND to UPTREND and sell when it changes from UPTREND to DOWNTREND.
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
        df.sort_values('Timestamp', inplace=True)
        df.set_index('Timestamp', inplace=True)
        
        # Print data preview and description for debugging/sharing purposes
        print("\n=== Log Data Preview (first 10 rows) ===")
        print(df.head(10))
        print("\n=== Log Data Description ===")
        print(df.describe(include='all'))

        # Create the plots directory if it doesn't exist
        plots_dir = os.path.join(LOGS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # ------------------------------
        # Optimal Trade Simulation
        # ------------------------------
        # We'll simulate optimal trades based on ideal trend reversals.
        # Initial portfolio: 500 USDC cash + TRUMP worth 500 USDC (Total: 1000)
        # For simulation, we assume we start in cash.
        initial_cash = 1000.0
        cash = initial_cash
        shares = 0.0
        optimal_values = []  # store portfolio value per timestamp
        optimal_signals = []  # to record simulation points (buy/sell)
        simulation_trend = df['Trend']
        prev_trend = simulation_trend.iloc[0]
        simulation_index = df.index

        # Iterate through each row in chronological order
        for i, (ts, row) in enumerate(df.iterrows()):
            price = row['Price']
            current_trend = row['Trend']
            # Check for trend reversals
            if i > 0:
                # DOWNTREND->UPTREND: ideal moment to BUY (convert all cash to asset)
                if prev_trend == "DOWNTREND" and current_trend == "UPTREND" and cash > 0:
                    shares = cash / price
                    cash = 0
                    optimal_signals.append((ts, price, "BUY"))
                # UPTREND->DOWNTREND: ideal moment to SELL (convert all asset to cash)
                elif prev_trend == "UPTREND" and current_trend == "DOWNTREND" and shares > 0:
                    cash = shares * price
                    shares = 0
                    optimal_signals.append((ts, price, "SELL"))
            # Compute portfolio value: if in asset, value = shares * current price; otherwise, cash.
            portfolio_value = cash if cash > 0 else shares * price
            optimal_values.append(portfolio_value)
            prev_trend = current_trend

        # Add the simulation curve into the dataframe for plotting
        df['Optimal_Portfolio_Value'] = optimal_values

        # ------------------------------
        # Create the plots
        # ------------------------------
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 18), sharex=True)
        fig.suptitle('Trading Performance and Optimal Trade Simulation', fontsize=18)

        # Plot 1: Market Price with Trend Shading
        axes[0].plot(df.index, df['Price'], label='Market Price', color='purple')
        axes[0].set_ylabel('Price')
        axes[0].set_title('Market Price and Trends')
        axes[0].grid(True)
        # Shade background by trend
        for i in range(len(df)-1):
            trend = df['Trend'].iloc[i]
            color = 'lightgreen' if trend == 'UPTREND' else 'lightcoral' if trend == 'DOWNTREND' else 'lightgray'
            axes[0].axvspan(df.index[i], df.index[i+1], color=color, alpha=0.3)
        axes[0].legend(loc='upper left')

        # Plot 2: Buy/Sell Signals (from generated signals in log)
        axes[1].plot(df.index, df['Price'], label='Price', color='purple')
        axes[1].set_ylabel('Price')
        axes[1].set_title('Generated Trade Signals')
        axes[1].grid(True)
        # Plot signals from log if any (even if they are NEUTRAL, we mark those with specific markers)
        buy_signals = df[df['Trade_Action'].str.upper() == 'BUY']
        sell_signals = df[df['Trade_Action'].str.upper() == 'SELL']
        axes[1].scatter(buy_signals.index, buy_signals['Price'], color='green', marker='^', s=100, label='BUY Signal')
        axes[1].scatter(sell_signals.index, sell_signals['Price'], color='red', marker='v', s=100, label='SELL Signal')
        axes[1].legend(loc='upper left')

        # Plot 3: Optimal Simulation Portfolio Value
        axes[2].plot(df.index, df['Optimal_Portfolio_Value'], label='Optimal Portfolio Value', color='orange', linewidth=2)
        axes[2].set_ylabel('Portfolio Value (USDC)')
        axes[2].set_title('Optimal Trade Simulation')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')
        # Annotate the buy/sell points from simulation
        for ts, price, action in optimal_signals:
            if action == "BUY":
                axes[2].annotate(f'BUY\n@{price:.2f}', xy=(ts, df.loc[ts, 'Optimal_Portfolio_Value']),
                                 xytext=(5, 15), textcoords='offset points', color='green', fontweight='bold')
            elif action == "SELL":
                axes[2].annotate(f'SELL\n@{price:.2f}', xy=(ts, df.loc[ts, 'Optimal_Portfolio_Value']),
                                 xytext=(5, -15), textcoords='offset points', color='red', fontweight='bold')

        # Plot 4: Comparison of Actual Portfolio Value vs. Optimal Simulation
        axes[3].plot(df.index, df['Portfolio_Value'], label='Actual Portfolio Value', color='blue', linestyle='--')
        axes[3].plot(df.index, df['Optimal_Portfolio_Value'], label='Optimal Portfolio Value', color='orange', linewidth=2)
        axes[3].set_ylabel('Portfolio Value (USDC)')
        axes[3].set_title('Actual vs. Optimal Portfolio Value')
        axes[3].grid(True)
        axes[3].legend(loc='upper left')

        # Format x-axis to show dates nicely
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot in logs folder
        plot_path = os.path.join(plots_dir, 'trading_performance_improved.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nImproved plot saved as: {plot_path}")
        
        # Display the plot
        plt.show()

        # Print some statistics
        print("\n=== Trading Statistics ===")
        total_trades = len(buy_signals) + len(sell_signals)
        print(f"Total Trades (from log): {total_trades}")
        print(f"Buy Signals: {len(buy_signals)}")
        print(f"Sell Signals: {len(sell_signals)}")
        
        try:
            df['Portfolio_Value'] = df['Portfolio_Value'].replace(['-', ''], '0')  # Replace '-' and empty values with '0'
            initial_value = float(df['Portfolio_Value'].iloc[0])
            final_value = float(df['Portfolio_Value'].iloc[-1])
            print(f"Initial Actual Portfolio Value: ${initial_value:.2f}")
            print(f"Final Actual Portfolio Value: ${final_value:.2f}")
            if initial_value > 0:
                print(f"Total Return (Actual): {((final_value / initial_value) - 1) * 100:.2f}%")
            else:
                print("Total Return (Actual): N/A (insufficient data)")
        except Exception as e:
            print("Could not calculate portfolio statistics due to invalid data")
            print(f"Error details: {e}")

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_filepath}'.")
        print("Please run the trend detector first to generate trading data.")
    except pd.errors.EmptyDataError:
        print(f"Error: Log file '{log_filepath}' is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_trading_log() 