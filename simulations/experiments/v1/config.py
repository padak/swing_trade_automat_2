"""
Experiment configuration V1: Aggressive RSI with tight MACD
This experiment tests more aggressive RSI levels with tighter MACD parameters.
"""

from simulations.base_config import BaseConfig

class ExperimentConfig(BaseConfig):
    # Override parameters for this experiment
    RSI_OVERBOUGHT = 75  # More aggressive overbought level
    RSI_OVERSOLD = 30    # More aggressive oversold level
    
    # Tighter MACD settings
    MACD_FAST_PERIOD = 8
    MACD_SLOW_PERIOD = 17
    MACD_SIGNAL_PERIOD = 9
    
    # Experiment Metadata
    NAME = "v1_aggressive_rsi"
    DESCRIPTION = "Testing aggressive RSI levels with tighter MACD parameters"
    VERSION = "1.0" 