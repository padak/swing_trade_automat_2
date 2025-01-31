# Swing Trading Automat

An automated trading system for Binance using Python.

## Project Structure

```
swing_trade_automat_2/
├── docs/               # Documentation and development plans
├── logs/              # Trading logs and performance plots
│   └── plots/         # Generated trading performance visualizations
├── model/             # Trained ML models
├── src/               # Source code
├── tools/             # Analysis and utility tools
├── venv/              # Python virtual environment
├── .env               # Environment variables (not in git)
└── requirements.txt   # Python dependencies
```

## Setup

1. Create required directories:
```bash
mkdir -p logs/plots model
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Binance API credentials:
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional trading parameters (defaults will be used if not set)
SYMBOL=TRUMPUSDC
INVEST_USD=500
PROFIT_THRESHOLD_PERCENT=0.002
SLEEP_TIME=300  # 5 minutes in seconds

# Strategy parameters (optional)
FAST_MA_PERIOD=10
SLOW_MA_PERIOD=30
RSI_PERIOD=14
RSI_OVERBOUGHT=70
RSI_OVERSOLD=35
```

## Running the System

1. Start the trading bot:
```bash
python src/trend_detector_v2_gemini.py
```

2. Analyze trading performance:
```bash
python tools/analyze_trading_log.py
```

3. View trading performance plots:
```bash
python tools/plot_log.py
```

4. Inspect the trained model:
```bash
python tools/inspect_model.py
```

## Directory Usage

- `logs/`: Contains trading activity logs and performance metrics
  - `plots/`: Generated visualizations of trading performance
- `model/`: Stores trained machine learning models
- `src/`: Main trading bot implementation
- `tools/`: Utility scripts for analysis and visualization
  - `analyze_trading_log.py`: Analyzes trading performance and suggests improvements
  - `plot_log.py`: Generates visual performance charts
  - `inspect_model.py`: Examines the trained ML model parameters

## Development

Follow the incremental development plan outlined in `docs/DEVELOPER_PLAN.md`.

## Monitoring

1. Check the trading log in `logs/trading_log.csv` for detailed trade history
2. View performance plots in `logs/plots/trading_performance.png`
3. Monitor model performance using the analysis tools in the `tools/` directory

## Troubleshooting

1. If you see "Model not found" errors:
   - Run the trading bot first to generate the initial model
   - Check that the `model/` directory exists

2. If you see "Log file not found" errors:
   - Ensure the trading bot has been run at least once
   - Verify the `logs/` directory exists

3. If plots are not generating:
   - Ensure the `logs/plots/` directory exists
   - Check that matplotlib is properly installed
   - Verify there is data in the trading log file 