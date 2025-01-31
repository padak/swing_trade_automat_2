# Trading Strategy Parameter Simulation

This document describes how to use the parameter simulation system to experiment with different trading strategy configurations.

## Overview

The simulation system allows you to test different trading parameters using historical market data. Each experiment is defined in a separate configuration file, making it easy to compare different strategies and parameter sets.

## Directory Structure

```
simulations/
├── __init__.py
├── base_config.py          # Base configuration with default parameters
├── experiment_manager.py   # Handles experiment execution and results
├── market_data/           # Historical market data for simulations
│   └── TRUMPUSDC.csv
└── experiments/           # Individual experiment configurations
    └── v1/
        ├── config.py      # Experiment-specific parameters
        └── results/       # Experiment results in JSON format
```

## Running Experiments

To run a simulation experiment:

```bash
python src/trend_detector_backtest.py --experiment simulations/experiments/v1
```

## Creating New Experiments

1. Create a new directory under `simulations/experiments/` (e.g., `v2/`)
2. Create a `config.py` file that inherits from `BaseConfig`
3. Override parameters you want to test

Example experiment configuration:

```python
from simulations.base_config import BaseConfig

class ExperimentConfig(BaseConfig):
    # Override parameters for this experiment
    RSI_OVERBOUGHT = 75
    RSI_OVERSOLD = 30
    MACD_FAST_PERIOD = 8
    
    # Metadata
    NAME = "v2_custom_strategy"
    DESCRIPTION = "Testing modified RSI levels"
    VERSION = "1.0"
```

## Available Parameters

### Trading Parameters
- `SYMBOL`: Trading pair symbol (default: "TRUMPUSDC")
- `INITIAL_USDC`: Initial USDC balance (default: 500)
- `INITIAL_TRUMP_USDC_VALUE`: Initial TRUMP value in USDC (default: 500)
- `MIN_TRADE_USDC`: Minimum trade size in USDC (default: 1.2)

### Technical Indicators
- `FAST_MA_PERIOD`: Fast Moving Average period (default: 10)
- `SLOW_MA_PERIOD`: Slow Moving Average period (default: 30)
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `RSI_OVERBOUGHT`: RSI overbought level (default: 70)
- `RSI_OVERSOLD`: RSI oversold level (default: 35)
- `MACD_FAST_PERIOD`: MACD fast period (default: 12)
- `MACD_SLOW_PERIOD`: MACD slow period (default: 26)
- `MACD_SIGNAL_PERIOD`: MACD signal period (default: 9)
- `BOLLINGER_BAND_PERIOD`: Bollinger Bands period (default: 20)
- `BOLLINGER_BAND_STD`: Bollinger Bands standard deviation (default: 2)

### Risk Management
- `STOP_LOSS_PERCENT`: Stop loss percentage (default: 0.02)
- `TAKE_PROFIT_PERCENT`: Take profit percentage (default: 0.10)
- `TRADE_RISK_PERCENT`: Risk per trade percentage (default: 0.01)

### Strategy Controls
- `MACD_CONFIRMATION`: Require MACD confirmation for signals (default: True)
- `MIN_PRICE_CHANGE`: Minimum price change for signal (default: 1.0)

## Results Analysis

Experiment results are saved in JSON format in the experiment's `results/` directory. Each result file contains:
- Trading statistics (total trades, win rate, etc.)
- Performance metrics (total return, benchmark comparison)
- Experiment parameters used
- Timestamp and metadata

To analyze results:
```python
from simulations.experiment_manager import ExperimentManager

# Load experiment results
experiment = ExperimentManager("simulations/experiments/v1")
results = experiment.get_all_results()

# Access specific result
latest_result = results[-1]
print(f"Win Rate: {latest_result['win_rate']}%")
print(f"Total Return: {latest_result['total_return']}%")
```

## Best Practices

1. Start with small parameter changes from the base configuration
2. Test one parameter group at a time
3. Document the hypothesis for each experiment in the config file
4. Compare results against the benchmark performance
5. Keep experiment configurations in version control 