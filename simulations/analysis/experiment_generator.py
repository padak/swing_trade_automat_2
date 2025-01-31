#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime
from typing import Dict, Optional
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import default parameters from trend detector script
from src.trend_detector_backtest import (
    FAST_MA_PERIOD_DEFAULT,
    SLOW_MA_PERIOD_DEFAULT,
    RSI_PERIOD_DEFAULT,
    RSI_OVERBOUGHT_DEFAULT,
    RSI_OVERSOLD_DEFAULT,
    MACD_FAST_PERIOD_DEFAULT,
    MACD_SLOW_PERIOD_DEFAULT,
    MACD_SIGNAL_PERIOD_DEFAULT,
    BOLLINGER_BAND_PERIOD_DEFAULT,
    BOLLINGER_BAND_STD_DEFAULT,
    STOP_LOSS_PERCENT_DEFAULT,
    TAKE_PROFIT_PERCENT_DEFAULT,
    TRADE_RISK_PERCENT_DEFAULT,
    MIN_PRICE_CHANGE_DEFAULT,
    MACD_CONFIRMATION_DEFAULT
)

class ExperimentGenerator:
    def __init__(self, base_experiment_path: str):
        """
        Initialize the experiment generator.
        
        Args:
            base_experiment_path: Path to the base experiment to use as a template
        """
        self.base_experiment_path = base_experiment_path
        self.base_config = self._load_base_config()

    def _load_base_config(self) -> dict:
        """Load the base configuration file."""
        config_path = os.path.join(self.base_experiment_path, "config.py")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.py found in {self.base_experiment_path}")
        
        # Read config.py as text
        with open(config_path, 'r') as f:
            config_text = f.read()
        
        # Create a new dictionary with base config values
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
        
        # Extract parameter values from config text
        try:
            # Create a copy of globals for execution context
            exec_globals = globals().copy()
            # Execute the config file with our params dictionary
            exec(config_text, exec_globals, params)
        except Exception as e:
            print(f"Warning: Error parsing config file: {e}")
            print("Using default parameters instead")
        
        return params

    def generate_new_experiment(self, 
                              new_params: Dict,
                              experiment_name: Optional[str] = None,
                              description: str = "") -> str:
        """
        Generate a new experiment configuration based on analysis recommendations.
        
        Args:
            new_params: Dictionary of new parameter values
            experiment_name: Optional name for the experiment
            description: Description of the experiment
            
        Returns:
            Path to the new experiment directory
        """
        # Generate experiment name if not provided
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"v{timestamp}"

        # Create new experiment directory
        experiments_dir = os.path.dirname(self.base_experiment_path)
        new_experiment_path = os.path.join(experiments_dir, experiment_name)
        os.makedirs(new_experiment_path, exist_ok=True)
        os.makedirs(os.path.join(new_experiment_path, "results"), exist_ok=True)

        # Merge base config with new parameters
        config = self.base_config.copy()
        config.update(new_params)

        # Generate config.py content
        config_content = [
            "# Auto-generated experiment configuration",
            f"# Generated at: {datetime.now().isoformat()}",
            f"# Based on: {self.base_experiment_path}",
            f"# Description: {description}",
            "",
            "from src.trend_detector_backtest import *  # Import default parameters",
            "from datetime import datetime",
            "",
            "class ExperimentConfig:",
            "    \"\"\"Configuration for the experiment.\"\"\"",
            "",
            f"    NAME = \"{experiment_name}\"  # Experiment name",
            f"    DESCRIPTION = \"{description}\"  # Experiment description",
            f"    VERSION = \"{experiment_name}\"  # Experiment version",
            f"    CREATED_AT = \"{datetime.now().isoformat()}\"  # Creation timestamp",
            "    TAGS = [\"automated\", \"parameter_optimization\"]  # Experiment tags",
            "",
            "    @classmethod",
            "    def get_metadata(cls):",
            "        \"\"\"Get experiment metadata.\"\"\"",
            "        return {",
            "            'name': cls.NAME,",
            "            'description': cls.DESCRIPTION,",
            "            'version': cls.VERSION,",
            "            'created_at': cls.CREATED_AT,",
            "            'tags': cls.TAGS",
            "        }",
            ""
        ]

        # Add parameters that differ from defaults
        default_values = {
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

        # First add a comment showing all default values
        config_content.append("    # Default values from trend_detector_backtest.py:")
        for key, value in default_values.items():
            config_content.append(f"    # {key} = {value}")
        config_content.append("")
        
        # Then add the overridden values
        config_content.append("    # Experiment-specific parameter values:")
        for key, value in config.items():
            if key in default_values and value != default_values[key]:
                if isinstance(value, str):
                    config_content.append(f"    {key} = \"{value}\"")
                else:
                    config_content.append(f"    {key} = {value}")
        
        # Add get_params method
        config_content.extend([
            "",
            "    @classmethod",
            "    def get_params(cls):",
            "        \"\"\"Get all parameters as a dictionary.\"\"\"",
            "        params = {",
            "            'FAST_MA_PERIOD': FAST_MA_PERIOD_DEFAULT,",
            "            'SLOW_MA_PERIOD': SLOW_MA_PERIOD_DEFAULT,",
            "            'RSI_PERIOD': RSI_PERIOD_DEFAULT,",
            "            'RSI_OVERBOUGHT': RSI_OVERBOUGHT_DEFAULT,",
            "            'RSI_OVERSOLD': RSI_OVERSOLD_DEFAULT,",
            "            'MACD_FAST_PERIOD': MACD_FAST_PERIOD_DEFAULT,",
            "            'MACD_SLOW_PERIOD': MACD_SLOW_PERIOD_DEFAULT,",
            "            'MACD_SIGNAL_PERIOD': MACD_SIGNAL_PERIOD_DEFAULT,",
            "            'BOLLINGER_BAND_PERIOD': BOLLINGER_BAND_PERIOD_DEFAULT,",
            "            'BOLLINGER_BAND_STD': BOLLINGER_BAND_STD_DEFAULT,",
            "            'STOP_LOSS_PERCENT': STOP_LOSS_PERCENT_DEFAULT,",
            "            'TAKE_PROFIT_PERCENT': TAKE_PROFIT_PERCENT_DEFAULT,",
            "            'TRADE_RISK_PERCENT': TRADE_RISK_PERCENT_DEFAULT,",
            "            'MIN_PRICE_CHANGE': MIN_PRICE_CHANGE_DEFAULT,",
            "            'MACD_CONFIRMATION': MACD_CONFIRMATION_DEFAULT",
            "        }"
        ])

        # Add overridden parameters to get_params method
        for key, value in config.items():
            if key in default_values and value != default_values[key]:
                if isinstance(value, str):
                    config_content.append(f"        params['{key}'] = \"{value}\"")
                else:
                    config_content.append(f"        params['{key}'] = {value}")
        
        config_content.append("        return params")

        # Write config.py
        config_path = os.path.join(new_experiment_path, "config.py")
        with open(config_path, 'w') as f:
            f.write('\n'.join(config_content))

        return new_experiment_path

def load_latest_analysis(experiment_path: str) -> Dict:
    """
    Load the most recent analysis results for an experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    results_dir = os.path.join(experiment_path, "results")
    analysis_files = [f for f in os.listdir(results_dir) if f.startswith('analysis_')]
    
    if not analysis_files:
        raise FileNotFoundError(f"No analysis files found in {results_dir}")
    
    # Use the most recent analysis file
    latest_analysis = sorted(analysis_files)[-1]
    analysis_path = os.path.join(results_dir, latest_analysis)
    
    with open(analysis_path, 'r') as f:
        return json.load(f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate new experiment configuration')
    parser.add_argument('--base-experiment', type=str, required=True,
                      help='Path to base experiment (e.g., simulations/experiments/v1)')
    parser.add_argument('--name', type=str, required=True,
                      help='Name for the new experiment (e.g., v2)')
    args = parser.parse_args()
    
    # Initialize generator with base experiment
    generator = ExperimentGenerator(args.base_experiment)
    
    # Load latest analysis results
    try:
        analysis = load_latest_analysis(args.base_experiment)
        print(f"Loaded analysis from {args.base_experiment}")
        
        # Get recommended parameters from analysis
        if 'recommendations' in analysis and 'new_experiment_params' in analysis['recommendations']:
            new_params = analysis['recommendations']['new_experiment_params']
            description = "Experiment with parameters optimized based on analysis recommendations"
        else:
            print("No parameter recommendations found in analysis, using default adjustments")
            new_params = {
                'RSI_OVERSOLD': 30,
                'RSI_OVERBOUGHT': 70,
                'MACD_FAST_PERIOD': 12,
                'MACD_SLOW_PERIOD': 26,
                'MACD_SIGNAL_PERIOD': 9,
                'MIN_PRICE_CHANGE': 1.5
            }
            description = "Experiment with default parameter adjustments"
            
        # Print analysis summary
        if 'trade_analysis' in analysis:
            print("\nPrevious experiment results:")
            print(f"Win rate: {analysis['trade_analysis'].get('win_rate', 'N/A')}%")
            print(f"Average profit per trade: ${analysis['trade_analysis'].get('avg_profit_per_trade', 'N/A')}")
        
        if 'recommendations' in analysis:
            print("\nRecommendations:")
            for adj in analysis['recommendations'].get('parameter_adjustments', []):
                print(f"- {adj}")
        
        # Generate new experiment
        new_experiment_path = generator.generate_new_experiment(
            new_params,
            experiment_name=args.name,
            description=description
        )
        
        print(f"\nGenerated new experiment at: {new_experiment_path}")
        print("New parameters:")
        for key, value in new_params.items():
            print(f"- {key}: {value}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run analyze_experiment.py first to generate analysis results")
        sys.exit(1)

if __name__ == "__main__":
    main() 