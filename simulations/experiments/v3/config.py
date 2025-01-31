# Auto-generated experiment configuration
# Generated at: 2025-01-31T14:57:32.356657
# Based on: simulations/experiments/v2
# Description: Experiment with parameters optimized based on analysis recommendations

from src.trend_detector_backtest import *  # Import default parameters
from datetime import datetime

class ExperimentConfig:
    """Configuration for the experiment."""

    NAME = "v3"  # Experiment name
    DESCRIPTION = "Experiment with parameters optimized based on analysis recommendations"  # Experiment description
    VERSION = "v3"  # Experiment version
    CREATED_AT = "2025-01-31T14:57:32.356689"  # Creation timestamp
    TAGS = ["automated", "parameter_optimization"]  # Experiment tags

    @classmethod
    def get_metadata(cls):
        """Get experiment metadata."""
        return {
            'name': cls.NAME,
            'description': cls.DESCRIPTION,
            'version': cls.VERSION,
            'created_at': cls.CREATED_AT,
            'tags': cls.TAGS
        }

    # Default values from trend_detector_backtest.py:
    # FAST_MA_PERIOD = 10
    # SLOW_MA_PERIOD = 30
    # RSI_PERIOD = 14
    # RSI_OVERBOUGHT = 70
    # RSI_OVERSOLD = 35
    # MACD_FAST_PERIOD = 12
    # MACD_SLOW_PERIOD = 26
    # MACD_SIGNAL_PERIOD = 9
    # BOLLINGER_BAND_PERIOD = 20
    # BOLLINGER_BAND_STD = 2
    # STOP_LOSS_PERCENT = 0.02
    # TAKE_PROFIT_PERCENT = 0.1
    # TRADE_RISK_PERCENT = 0.01
    # MIN_PRICE_CHANGE = 1.0
    # MACD_CONFIRMATION = True

    # Experiment-specific parameter values:

    @classmethod
    def get_params(cls):
        """Get all parameters as a dictionary."""
        params = {
            'FAST_MA_PERIOD': FAST_MA_PERIOD_DEFAULT,
            'SLOW_MA_PERIOD': SLOW_MA_PERIOD_DEFAULT,
            'RSI_PERIOD': RSI_PERIOD_DEFAULT,
            'RSI_OVERBOUGHT': RSI_OVERBOUGHT_DEFAULT,
            'RSI_OVERSOLD': RSI_OVERSOLD_DEFAULT,
            'MACD_FAST_PERIOD': MACD_FAST_PERIOD_DEFAULT,
            'MACD_SLOW_PERIOD': MACD_SLOW_PERIOD_DEFAULT,
            'MACD_SIGNAL_PERIOD': MACD_SIGNAL_PERIOD_DEFAULT,
            'BOLLINGER_BAND_PERIOD': BOLLINGER_BAND_PERIOD_DEFAULT,
            'BOLLINGER_BAND_STD': BOLLINGER_BAND_STD_DEFAULT,
            'STOP_LOSS_PERCENT': STOP_LOSS_PERCENT_DEFAULT,
            'TAKE_PROFIT_PERCENT': TAKE_PROFIT_PERCENT_DEFAULT,
            'TRADE_RISK_PERCENT': TRADE_RISK_PERCENT_DEFAULT,
            'MIN_PRICE_CHANGE': MIN_PRICE_CHANGE_DEFAULT,
            'MACD_CONFIRMATION': MACD_CONFIRMATION_DEFAULT
        }
        return params