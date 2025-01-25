# Trend Detector v1 Gemini - Self-Learning Trading System for Binance

This document describes the `trend_detector_v1_gemini.py` script, a self-learning cryptocurrency trading system designed to simulate trading on Binance. It is built for end-users who want an automated trading solution that optimizes itself over time without requiring manual parameter adjustments.

## Overview

The Trend Detector v1 Gemini system uses a combination of technical indicators and a self-learning mechanism to make trading decisions for the TRUMP/USDC pair on Binance. It starts with an initial capital of 500 USDC and 500 USDC worth of TRUMP coins. The system continuously monitors the market, generates trading signals, executes simulated trades, and adjusts its trading strategy based on its performance.

**Key Features:**

*   **Self-Learning:** The system automatically adjusts its trading parameters based on past performance, aiming to optimize profitability over time.
*   **Technical Indicators:** It uses Moving Averages (MA) and Relative Strength Index (RSI) to identify potential trading opportunities.
*   **Simulated Trading:** The script simulates trading using the Binance API to fetch real-time market data but does not execute real trades (in this version). You can adapt it for real trading by implementing actual order placement using the Binance SDK. **Use with caution and at your own risk for real trading.**
*   **Automated Operation:** Once started, the system runs continuously, making trading decisions without manual intervention.
*   **Parameter Optimization:** The system starts with default parameters and iteratively refines them to improve trading performance.
*   **Risk Management:** Includes basic risk management with a minimum trade size and parameter adjustments to become more conservative during poor performance.

## Trend Detector Script Logic Flow

This section describes the logical flow of the `trend_detector_v1_gemini.py` script to help understand its inner workings.

```mermaid
graph TB
    subgraph Initialization
        A[Start] --> B[Load API Keys & Secrets]
        B --> C[Set Initial Parameters]
        C --> D[Initialize Binance Client]
        D -- Success --> E[Calculate Initial TRUMP Quantity]
        D -- Failure --> Fail[Initialization Failed - Exit]
        E -- Price Fetched --> F[Fetch Initial Historical Data for ML]
        E -- No Price --> Fail2[Could Not Fetch Initial Price - Exit]
        F -- Data Fetched --> G[Initial Balances Set]
        F -- No Data --> Warn[Warning: No Historical Data for ML]
        G --> H[Initial Strategy Parameters Set]
        H --> ML_Start[Start ML Model Training Optional]
    end

    subgraph ML_Model_Training
        ML_Start --> ML_Fetch[Fetch Historical Data for Training]
        ML_Fetch -- Data Fetched --> ML_Features[Calculate Features MA RSI Price Change]
        ML_Fetch -- No Data --> ML_NoData[ML Training Skipped]
        ML_Features --> ML_Split[Split Data Train/Test]
        ML_Split --> ML_Train[Train Logistic Regression Model]
        ML_Train -- Success --> ML_Eval[Evaluate Model Accuracy]
        ML_Train -- Failure --> ML_Fail[ML Training Failed - Continue without ML]
        ML_Eval --> ML_End[ML Model Trained & Evaluated]
        ML_Fail --> ML_End
        ML_NoData --> ML_End
    end

    subgraph Trading_Loop
        ML_End --> Trading_Start[Start Trading Loop]
        Trading_Start --> Iteration_Start[Iteration Start]
        Iteration_Start --> FetchPrice[Fetch Current Price]
        FetchPrice -- Price Fetched --> PriceHistory[Append Price to History]
        FetchPrice -- No Price --> WaitRetry[Wait & Retry]
        WaitRetry --> Iteration_Start
        PriceHistory --> DataCheck[Enough Data for Indicators?]
        DataCheck -- Yes --> SignalGen_Start[Signal Generation]
        DataCheck -- No --> PerfEval_NoSignal[Performance Evaluation No Signal]

        subgraph Signal_Generation
            SignalGen_Start --> RuleBasedSig[Generate Rule-Based Signal]
            RuleBasedSig --> ML_Model_Check[Is ML Model Trained?]
            ML_Model_Check -- Yes --> ML_SignalGen[Generate ML Signal]
            ML_Model_Check -- No --> CombineSignals[Use Rule-Based Signal]
            ML_SignalGen --> CombineSignals[Combine Signals]
            CombineSignals --> TradeDecision[Trading Signal Decision]
        end

        TradeDecision -- BUY/SELL --> ExecuteTrade[Execute Trade Simulated]
        TradeDecision -- NEUTRAL --> PerfEval_NoSignal

        subgraph Trade_Execution
            ExecuteTrade --> MinTradeCheck[Check Minimum Trade Quantity]
            MinTradeCheck -- Met --> TradeExecutionLogic[Trade Execution Logic]
            MinTradeCheck -- Not Met --> NoTrade[No Trade - Below Minimum]
            TradeExecutionLogic -- BUY --> BuyOrder[Simulate BUY Order]
            TradeExecutionLogic -- SELL --> SellOrder[Simulate SELL Order]
            BuyOrder --> BalanceUpdate[Update Balances]
            SellOrder --> BalanceUpdate
            NoTrade --> BalanceUpdate
            BalanceUpdate --> PerfEval_Signal
        end

        PerfEval_Signal[Performance Evaluation After Signal] --> PortfolioEval
        PerfEval_NoSignal --> PortfolioEval

        PortfolioEval[Evaluate Portfolio Value & P/L] --> StorePerformance[Store Performance History]
        StorePerformance --> ParameterAdjCheck[Check for Parameter Adjustment Interval]
        ParameterAdjCheck -- Yes --> ParameterAdjust[Adjust Strategy Parameters]
        ParameterAdjCheck -- No --> LoopEnd[Loop End - Wait]
        ParameterAdjust --> LoopEnd
        LoopEnd --> WaitTime[Wait Time]
        WaitTime --> CheckTradingEnabled[Check if Trading Enabled]
        CheckTradingEnabled -- Yes --> Iteration_Start
        CheckTradingEnabled -- No --> Trading_End[Trading Loop End - Exit]
    end

    Trading_End --> Cleanup[Cleanup Optional]
    Cleanup --> Z[End]
    Fail --> Z
    Fail2 --> Z
    Warn --> Trading_Start
    ML_NoData --> Trading_Start
    ML_Fail --> Trading_Start
```

The flowchart above illustrates the following steps:

1.  **Start Script**: The script execution begins.
2.  **Load Configuration**:  The script loads configurations from a file or environment variables, including data source details and trend detection method.
3.  **Check Data Source**: The script determines the source of the time series data (e.g., CSV file or InfluxDB).
4.  **Read Data**: Based on the data source, the script reads data either from a CSV file or by querying InfluxDB.
5.  **Data Preprocessing**: The loaded data is preprocessed, which might include cleaning, formatting, and handling missing values.
6.  **Select Trend Detection Method**: The script chooses a trend detection method based on the configuration (e.g., Simple Moving Average or Linear Regression).
7.  **Apply Trend Detection Method**: The selected method is applied to the preprocessed data to identify trends.
8.  **Analyze Trend Results**: The results from the trend detection method are analyzed to determine the presence and characteristics of trends.
9.  **Output Trend Report**: A report summarizing the detected trends is generated.
10. **Display/Save Report**: The report is either displayed to the user or saved to a file, depending on the script's configuration.
11. **End Script**: The script execution completes.

This visual representation should provide a clearer understanding of the script's logic and decision-making process.

## Strategy

The trading strategy is based on the following technical indicators:

1.  **Moving Averages (MA):**
    *   **Fast Moving Average:**  A shorter period MA to identify short-term trends.
    *   **Slow Moving Average:** A longer period MA to identify longer-term trends.
    *   **Crossover Strategy:**  A "BUY" signal is generated when the fast MA crosses above the slow MA, indicating a potential upward trend. A "SELL" signal is generated when the fast MA crosses below the slow MA, indicating a potential downward trend.

2.  **Relative Strength Index (RSI):**
    *   **RSI Period:**  Used to calculate the RSI, which measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
    *   **Overbought/Oversold Levels:**
        *   **Overbought:**  When RSI exceeds a certain level (e.g., 70), it suggests the asset may be overvalued and prone to a price decrease.
        *   **Oversold:**  When RSI falls below a certain level (e.g., 30), it suggests the asset may be undervalued and prone to a price increase.

**Trading Signal Generation:**

*   **BUY Signal:** Generated when the fast MA is above the slow MA AND the RSI is below the oversold level.
*   **SELL Signal:** Generated when the fast MA is below the slow MA AND the RSI is above the overbought level.
*   **NEUTRAL Signal:** In all other conditions, no trade is executed.

**Self-Learning Mechanism:**

The system evaluates its performance periodically (every 10 iterations in the current script) and adjusts its trading parameters based on the average profit/loss percentage over the recent period.

*   **Improved Performance (Profit > 0.5% average):** The system becomes more exploratory by:
    *   Decreasing the fast MA period (making it more sensitive to short-term changes).
    *   Increasing the RSI overbought level and decreasing the oversold level (widening the neutral RSI range).

*   **Poor Performance (Loss < -1.0% average):** The system becomes more conservative by:
    *   Increasing the slow MA period (making it less sensitive to short-term fluctuations).
    *   Decreasing the RSI overbought level and increasing the oversold level (narrowing the neutral RSI range).
    *   Widening the stop-loss percentage.

*   **Moderate Performance:** Minor adjustments are made to fine-tune the parameters.

**Initial Parameters:**

The script starts with the following initial parameters, which are then adjusted by the self-learning mechanism:

*   `FAST_MA_PERIOD = 5`: **Fast Moving Average Period.** This parameter defines the period for the fast moving average calculation. A shorter period (like 5) makes the moving average more sensitive to recent price changes, helping to identify short-term trends.
*   `SLOW_MA_PERIOD = 20`: **Slow Moving Average Period.** This parameter defines the period for the slow moving average. A longer period (like 20) makes the moving average smoother and less sensitive to short-term fluctuations, representing longer-term trends.
*   `RSI_PERIOD = 14`: **RSI Period.** This is the period used to calculate the Relative Strength Index (RSI). It determines the number of past periods used to measure the speed and change of price movements. A common value is 14.
*   `RSI_OVERBOUGHT = 70`: **RSI Overbought Level.** This is the threshold above which the RSI is considered to indicate that an asset is overbought, suggesting it might be overvalued and could experience a price decline.
*   `RSI_OVERSOLD = 30`: **RSI Oversold Level.** This is the threshold below which the RSI is considered to indicate that an asset is oversold, suggesting it might be undervalued and could experience a price increase.
*   `STOP_LOSS_PERCENT = 0.001`: **Stop Loss Percentage.** This parameter defines the percentage below the purchase price at which a stop-loss order would be triggered to limit potential losses on a trade. For example, 0.001 represents 0.1%.
*   `TAKE_PROFIT_PERCENT = 0.10`: **Take Profit Percentage.** This parameter defines the percentage above the purchase price at which a take-profit order would be triggered to secure profits on a trade. For example, 0.10 represents 10%.

## How to Use

1.  **Install Binance Python SDK:**
    ```bash
    pip install python-binance
    ```

2.  **Set up Binance API Keys:**
    *   Create API keys on your Binance account ([https://www.binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management)).
    *   **Important:** For simulated trading and testing, consider using the Binance Testnet API if you want to avoid any risk to your real funds. However, this script in its current form is designed for simulation using live data, not testnet. To use real trading, you would need to modify the `execute_trade` function to place actual orders. **Be extremely cautious when connecting to the real Binance API and trading with real funds.**
    *   Set environment variables `binance_api` and `binance_secret` with your API key and secret. For example, in your `.bashrc` or `.zshrc`:
        ```bash
        export binance_api="YOUR_API_KEY"
        export binance_secret="YOUR_API_SECRET"
        ```
        Replace `YOUR_API_KEY` and `YOUR_API_SECRET` with your actual API key and secret.

3.  **Run the Script:**
    ```bash
    python src/trend_detector_v1_gemini.py
    ```

4.  **Monitor the Output:** The script will print the trading process, including:
    *   Current price of TRUMP/USDC
    *   Trading signals (BUY, SELL, NEUTRAL)
    *   Simulated trade executions
    *   USDC and TRUMP balances
    *   Portfolio value and profit/loss percentage
    *   Adjustments to strategy parameters

## Risk Disclaimer

**This script is for educational and simulation purposes only.** Trading cryptocurrency involves significant risk of loss.

*   **Simulated Trading:** This script simulates trades and does not execute real orders on Binance in its current form. To use it for real trading, you would need to modify the `execute_trade` function to place actual orders using the Binance API.
*   **No Financial Advice:** This script and documentation are not financial advice.
*   **Use at Your Own Risk:** If you decide to trade with real funds, start with a very small amount and thoroughly test and understand the system before increasing your investment.
*   **API Security:** Protect your Binance API keys and secrets. Do not share them and ensure they are securely stored as environment variables or in a secure configuration.

## Further Improvements

*   **More Sophisticated Strategies:** Implement more advanced trading strategies, such as:
    *   Adding more technical indicators (e.g., MACD, Bollinger Bands).
    *   Using more complex signal combinations and conditions.
    *   Incorporating order book data or other market information.
*   **Advanced Self-Learning:**
    *   Implement more sophisticated optimization algorithms (e.g., gradient descent, genetic algorithms) for parameter tuning.
    *   Explore reinforcement learning techniques to train the trading strategy.
*   **Risk Management Enhancements:**
    *   Implement stop-loss and take-profit orders in the simulated trades. (Basic stop-loss/take-profit percentages are included, but not order execution logic).
    *   Dynamically adjust trade size based on risk and portfolio value.
*   **Backtesting:**
    *   Incorporate backtesting functionality to evaluate strategy performance on historical data.
*   **User Interface:**
    *   Develop a user interface (e.g., web-based or command-line) to monitor the system and configure parameters.
*   **Real Trading Implementation:**
    *   Implement the actual order placement logic using the Binance API to enable real trading (with extreme caution and thorough testing).
*   **Error Handling and Logging:**
    *   Improve error handling and implement more comprehensive logging for debugging and monitoring.
*   **Parameter Tuning Range Limits:**
    *   Add limits to the parameter tuning ranges to prevent parameters from going to extreme or unrealistic values.

This documentation provides a comprehensive guide to understanding and using the Trend Detector v1 Gemini system. Remember to use it responsibly and understand the risks involved in cryptocurrency trading.

## Data Fetching and Processing

To make trading decisions, the script fetches and processes market data from Binance at two main stages: **Initialization** and during each **Iteration** of the trading loop.

### 1. Initialization Data Fetching

When the script starts, it performs the following data fetching operations:

*   **Fetching Current Price for Initial Quantity Calculation:**
    *   The script first calls the `fetch_current_price(SYMBOL)` function to get the current price of the trading pair (e.g., `TRUMPUSDC`).
    *   **Example:** If the current price of `TRUMPUSDC` is $55.00, this function will return `55.00`.
    *   This price is used to calculate the initial quantity of `TRUMP` coins to hold, based on the `INITIAL_TRUMP_USDC_VALUE`.

*   **Fetching Historical Data for ML Model Training (Optional):**
    *   If the Machine Learning (ML) model training is enabled (which it is by default in the script), the script calls `fetch_binance_data(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=10000)`.
    *   This function fetches historical candlestick data for the `TRUMPUSDC` pair from Binance.
        *   `symbol=SYMBOL`: Specifies the trading pair, e.g., "TRUMPUSDC".
        *   `interval=Client.KLINE_INTERVAL_1HOUR`:  Specifies the candlestick interval, in this case, 1-hour candles.
        *   `limit=10000`:  Specifies the number of historical data points to fetch, here it's 10,000 hourly candles.
    *   **Example of Historical Data:** The fetched data is a Pandas DataFrame containing columns like: `timestamp`, `open`, `high`, `low`, `close`, `volume`, etc. For ML training, the script primarily uses the `close` prices.  A snippet of the historical data might look like this:

    ```
                        close
    timestamp
    2024-01-01 00:00:00  54.50
    2024-01-01 01:00:00  54.75
    2024-01-01 02:00:00  55.20
    ...
    2024-02-02 10:00:00  56.10
    ```
    *   This historical data is then used to:
        *   Calculate technical indicators like Moving Averages (MA) and Relative Strength Index (RSI) over historical periods.
        *   Prepare features and labels to train the Logistic Regression Machine Learning model.
        *   Train the ML model to predict trading signals based on historical patterns of MAs, RSI, and price changes.

### 2. Iteration Data Fetching (Trading Loop)

Inside the main trading loop (`while trading_enabled:`), in each iteration, the script fetches **only the following**:

*   **Current Price for Trading Decisions:**
    *   In each iteration, `fetch_current_price(SYMBOL)` is called to get the most up-to-date price of `TRUMPUSDC`.
    *   **Example:** If in iteration 1, the current price is $55.25, and in iteration 2 (after a short delay), the price becomes $55.30, the script will fetch and use these updated prices in each iteration.
    *   **Important:**  The script does *not* fetch historical kline data or order book information in each iteration.

### 3. Data Processing in Trading Loop Iterations

With the current price fetched in each iteration, and using the accumulated `prices_history`, the script performs the following processing steps **internally**:

1.  **Append to Price History:** The `current_price` is appended to the `prices_history` list, maintaining a history of recent prices. This list is used for internal calculations of technical indicators.
2.  **Generate Trading Signals:**
    *   **Rule-Based Signal:**  `generate_trading_signal()` function uses the *internally maintained* `prices_history` list, along with the current strategy parameters, to calculate:
        *   **Moving Averages (MA):** Fast MA and Slow MA are calculated using the last `FAST_MA_PERIOD` and `SLOW_MA_PERIOD` prices from the `prices_history`, respectively.  *No new historical data is fetched from Binance for this calculation in each iteration.*
        *   **Relative Strength Index (RSI):** RSI is calculated using the price changes within the `RSI_PERIOD` in the *`prices_history`*.  *Again, no new data is fetched from Binance for this RSI calculation in each iteration.*
        *   Based on these internally calculated MAs and RSI, a "BUY", "SELL", or "NEUTRAL" signal is generated.
    *   **ML Model Signal (If Model is Trained):** If the ML model was successfully trained during initialization, `predict_signal(model, current_features)` is used. This function takes the latest calculated features (MA, RSI, Price Change based on recent `prices_history`) and uses the trained ML model to predict a trading signal ("BUY", "SELL", or "NEUTRAL").
    *   **Signal Combination:** The script combines the rule-based and ML signals (if available) to determine the final trading `signal`. In the current logic, if the ML signal is not "NEUTRAL", it prioritizes the ML signal; otherwise, it uses the rule-based signal.
3.  **Execute Trade (Simulated):**
    *   `execute_trade(signal, current_price, usdc_balance, trump_balance)` simulates a trade based on the combined `signal`.
    *   If the signal is "BUY" or "SELL" and the trade quantity meets the minimum trade value, it simulates the trade execution at the `current_price`.
4.  **Update Balances:**
    *   `update_balance(trade_action, trade_qty, current_price, usdc_balance, trump_balance, trade_details)` updates the simulated `usdc_balance` and `trump_balance` based on the executed trade (BUY or SELL) and records the trade details.
5.  **Evaluate Performance:**
    *   `evaluate_performance(initial_portfolio_value, current_portfolio_value)` calculates the current portfolio value and the profit/loss percentage since the beginning.
6.  **Adjust Strategy Parameters (Periodically):**
    *   Every 10 iterations, `adjust_strategy_parameters(avg_performance)` is called. This function looks at the average performance over the last 10 iterations and adjusts the trading strategy parameters (`FAST_MA_PERIOD`, `SLOW_MA_PERIOD`, `RSI_OVERBOUGHT`, `RSI_OVERSOLD`, `STOP_LOSS_PERCENT`) to try and improve future performance.

**Data Not Fetched in Each Iteration:**

*   **Historical Kline Data:**  The script does *not* re-fetch historical candlestick data from Binance in each iteration after the initial optional fetch for ML training.
*   **Order Book Data:**  Order book depth or order book snapshots are not fetched.
*   **Volume Data:**  Real-time volume data (beyond what might be implicitly reflected in price changes) is not explicitly fetched in each iteration.
*   **Other Market Data:**  Any other real-time market data beyond the current price is not used in the current script's iteration logic.

By fetching only the current price and performing technical indicator calculations internally using the accumulated price history, the script maintains efficiency while implementing a trend-following strategy.

By fetching the current price in each iteration and using the historical price data and technical indicators, the script continuously monitors the market, makes trading decisions, and adapts its strategy over time. 