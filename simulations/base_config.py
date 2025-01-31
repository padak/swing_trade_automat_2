"""
Base configuration for trading experiments.
This file contains default parameters that can be overridden by specific experiment configurations.
"""

class BaseConfig:
    # Trading Parameters
    SYMBOL = "TRUMPUSDC"
    INITIAL_USDC = 500
    INITIAL_TRUMP_USDC_VALUE = 500
    MIN_TRADE_USDC = 1.2

    # Technical Indicators
    FAST_MA_PERIOD = 10
    SLOW_MA_PERIOD = 30
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 35
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    BOLLINGER_BAND_PERIOD = 20
    BOLLINGER_BAND_STD = 2

    # Risk Management
    STOP_LOSS_PERCENT = 0.02
    TAKE_PROFIT_PERCENT = 0.10
    TRADE_RISK_PERCENT = 0.01

    # Strategy Controls
    MACD_CONFIRMATION = True
    MIN_PRICE_CHANGE = 1.0

    # Experiment Metadata
    NAME = "base"
    DESCRIPTION = "Base configuration with default parameters"
    VERSION = "1.0"

    @classmethod
    def get_params(cls):
        """Get all parameters as a dictionary."""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 