#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse

class ExperimentAnalyzer:
    def __init__(self, experiment_path: str, market_data_path: Optional[str] = None):
        """
        Initialize the analyzer with paths to experiment data and market data.
        
        Args:
            experiment_path: Path to the experiment directory
            market_data_path: Optional path to market data file
        """
        self.experiment_path = experiment_path
        self.market_data_path = market_data_path
        self.results_dir = os.path.join(experiment_path, "results")
        
        # Load experiment data
        self.detailed_log = self._load_detailed_log()
        self.final_results = self._load_final_results()
        self.market_data = self._load_market_data() if market_data_path else None
        
        # Analysis results
        self.trade_analysis = {}
        self.signal_analysis = {}
        self.parameter_impact = {}
        self.market_context = {}
        
    def _load_detailed_log(self) -> pd.DataFrame:
        """Load and parse the detailed log file."""
        log_files = [f for f in os.listdir(self.results_dir) if f.startswith('detailed_log_')]
        if not log_files:
            raise FileNotFoundError("No detailed log files found")
        
        # Use the most recent log file
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join(self.results_dir, latest_log)
        
        # Read JSONL file
        data = []
        with open(log_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        return pd.DataFrame(data)

    def _load_final_results(self) -> dict:
        """Load the final results JSON file."""
        results_files = [f for f in os.listdir(self.results_dir) if f.startswith('results_')]
        if not results_files:
            raise FileNotFoundError("No results files found")
        
        # Use the most recent results file
        latest_results = sorted(results_files)[-1]
        results_path = os.path.join(self.results_dir, latest_results)
        
        with open(results_path, 'r') as f:
            return json.load(f)

    def _load_market_data(self) -> pd.DataFrame:
        """Load market data from CSV file."""
        return pd.read_csv(self.market_data_path)

    def analyze_trades(self) -> Dict:
        """
        Analyze individual trades and their performance.
        Returns a dictionary with trade analysis results.
        """
        trades_data = self.detailed_log[self.detailed_log['trade_action'].isin(['BUY', 'SELL'])]
        
        analysis = {
            'total_trades': len(trades_data),
            'buy_trades': len(trades_data[trades_data['trade_action'] == 'BUY']),
            'sell_trades': len(trades_data[trades_data['trade_action'] == 'SELL']),
            'trade_details': [],
            'profit_distribution': {},
            'hold_times': [],
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'best_trade': None,
            'worst_trade': None
        }

        # Process each trade
        current_position = None
        for _, trade in trades_data.iterrows():
            if trade['trade_action'] == 'BUY':
                current_position = {
                    'entry_time': pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': trade['price'],
                    'quantity': trade['trade_quantity'],
                    'entry_indicators': {
                        'rsi': trade['rsi'],
                        'macd': trade['macd'],
                        'macd_signal': trade['macd_signal'],
                        'ma_fast': trade['ma_fast'],
                        'ma_slow': trade['ma_slow']
                    }
                }
            elif trade['trade_action'] == 'SELL' and current_position:
                # Calculate trade metrics
                exit_time = pd.to_datetime(trade['timestamp'])
                entry_time = datetime.strptime(current_position['entry_time'], '%Y-%m-%d %H:%M:%S')
                hold_time = (exit_time - pd.to_datetime(entry_time)).total_seconds() / 3600  # hours
                profit = (trade['price'] - current_position['entry_price']) * current_position['quantity']
                profit_percent = (trade['price'] / current_position['entry_price'] - 1) * 100

                trade_detail = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': current_position['entry_price'],
                    'exit_price': trade['price'],
                    'quantity': current_position['quantity'],
                    'hold_time_hours': float(hold_time),  # Convert numpy float to Python float
                    'profit_usd': float(profit),  # Convert numpy float to Python float
                    'profit_percent': float(profit_percent),  # Convert numpy float to Python float
                    'entry_indicators': current_position['entry_indicators'],
                    'exit_indicators': {
                        'rsi': float(trade['rsi']),  # Convert numpy float to Python float
                        'macd': float(trade['macd']),
                        'macd_signal': float(trade['macd_signal']),
                        'ma_fast': float(trade['ma_fast']),
                        'ma_slow': float(trade['ma_slow'])
                    }
                }
                analysis['trade_details'].append(trade_detail)
                analysis['hold_times'].append(hold_time)

                # Update best/worst trades
                if not analysis['best_trade'] or profit > analysis['best_trade']['profit_usd']:
                    analysis['best_trade'] = trade_detail
                if not analysis['worst_trade'] or profit < analysis['worst_trade']['profit_usd']:
                    analysis['worst_trade'] = trade_detail

                current_position = None

        if analysis['trade_details']:
            # Calculate aggregate statistics
            profits = [t['profit_usd'] for t in analysis['trade_details']]
            analysis['win_rate'] = float(len([p for p in profits if p > 0]) / len(profits) * 100)
            analysis['avg_profit_per_trade'] = float(np.mean(profits))
            analysis['profit_distribution'] = {
                'mean': float(np.mean(profits)),
                'median': float(np.median(profits)),
                'std': float(np.std(profits)),
                'min': float(np.min(profits)),
                'max': float(np.max(profits))
            }

        self.trade_analysis = analysis
        return analysis

    def analyze_signals(self) -> Dict:
        """
        Analyze trading signals and their effectiveness.
        Returns a dictionary with signal analysis results.
        """
        signals_data = self.detailed_log[self.detailed_log['signal'].isin(['BUY', 'SELL'])]
        
        analysis = {
            'total_signals': int(len(signals_data)),
            'buy_signals': int(len(signals_data[signals_data['signal'] == 'BUY'])),
            'sell_signals': int(len(signals_data[signals_data['signal'] == 'SELL'])),
            'executed_signals': int(len(signals_data[signals_data['trade_action'].isin(['BUY', 'SELL'])])),
            'missed_signals': int(len(signals_data[signals_data['trade_action'] == 'NEUTRAL'])),
            'signal_execution_rate': 0.0,
            'signal_quality': {
                'true_positives': 0,
                'false_positives': 0,
                'accuracy': 0.0
            }
        }

        # Calculate signal execution rate
        if analysis['total_signals'] > 0:
            analysis['signal_execution_rate'] = float((analysis['executed_signals'] / analysis['total_signals']) * 100)

        self.signal_analysis = analysis
        return analysis

    def generate_recommendations(self) -> Dict:
        """
        Generate strategy improvement recommendations based on analysis.
        Returns a dictionary with recommendations.
        """
        recommendations = {
            'parameter_adjustments': [],
            'strategy_improvements': [],
            'new_experiment_params': {}
        }

        # Analyze trade profitability patterns
        if self.trade_analysis:
            # Check RSI patterns
            profitable_trades = [t for t in self.trade_analysis['trade_details'] if t['profit_usd'] > 0]
            if profitable_trades:
                entry_rsi_values = [t['entry_indicators']['rsi'] for t in profitable_trades]
                avg_entry_rsi = np.mean(entry_rsi_values)
                
                # RSI recommendations
                if avg_entry_rsi < 30:
                    recommendations['parameter_adjustments'].append(
                        "Consider lowering RSI_OVERSOLD threshold for better entry points"
                    )
                elif avg_entry_rsi > 70:
                    recommendations['parameter_adjustments'].append(
                        "Consider raising RSI_OVERBOUGHT threshold for better exit points"
                    )

            # Check hold times
            avg_hold_time = np.mean(self.trade_analysis['hold_times'])
            if avg_hold_time < 1:  # Less than 1 hour
                recommendations['strategy_improvements'].append(
                    "Consider increasing minimum hold time to avoid quick reversals"
                )
            elif avg_hold_time > 48:  # More than 48 hours
                recommendations['strategy_improvements'].append(
                    "Consider adding shorter-term profit taking rules"
                )

        # Generate new experiment parameters
        if self.trade_analysis['win_rate'] < 50:
            recommendations['new_experiment_params'] = {
                'RSI_OVERSOLD': 30,  # More conservative entry
                'RSI_OVERBOUGHT': 70,
                'MACD_FAST_PERIOD': 12,
                'MACD_SLOW_PERIOD': 26,
                'MACD_SIGNAL_PERIOD': 9,
                'MIN_PRICE_CHANGE': 1.5  # Increase minimum price change requirement
            }

        return recommendations

    def plot_trades(self, save_path: Optional[str] = None):
        """
        Create a visualization of trades on price chart.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot price
        plt.plot(pd.to_datetime(self.detailed_log['timestamp']), 
                self.detailed_log['price'], 
                label='Price', 
                color='gray', 
                alpha=0.6)
        
        # Plot buy points
        buy_points = self.detailed_log[self.detailed_log['trade_action'] == 'BUY']
        plt.scatter(pd.to_datetime(buy_points['timestamp']), 
                   buy_points['price'], 
                   color='green', 
                   marker='^', 
                   label='Buy', 
                   s=100)
        
        # Plot sell points
        sell_points = self.detailed_log[self.detailed_log['trade_action'] == 'SELL']
        plt.scatter(pd.to_datetime(sell_points['timestamp']), 
                   sell_points['price'], 
                   color='red', 
                   marker='v', 
                   label='Sell', 
                   s=100)
        
        plt.title('Trading Activity Overview')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze trading experiment results')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--market-data', type=str, required=False,
                       help='Path to market data file (optional)')
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.experiment, args.market_data)

    # Run analysis
    trade_analysis = analyzer.analyze_trades()
    signal_analysis = analyzer.analyze_signals()
    recommendations = analyzer.generate_recommendations()

    # Create analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.experiment, "results", f"analysis_{timestamp}.json")
    
    report = {
        "trade_analysis": trade_analysis,
        "signal_analysis": signal_analysis,
        "recommendations": recommendations
    }

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate and save plot
    plot_path = os.path.join(args.experiment, "results", f"trades_plot_{timestamp}.png")
    analyzer.plot_trades(save_path=plot_path)

    print(f"\nAnalysis completed. Report saved to: {report_path}")
    print(f"Trade plot saved to: {plot_path}")

    # Print key findings
    print("\nKey Findings:")
    print(f"Total Trades: {trade_analysis['total_trades']}")
    print(f"Win Rate: {trade_analysis['win_rate']:.2f}%")
    print(f"Average Profit per Trade: ${trade_analysis['avg_profit_per_trade']:.2f}")
    print(f"Signal Execution Rate: {signal_analysis['signal_execution_rate']:.2f}%")
    
    print("\nRecommendations:")
    for adj in recommendations['parameter_adjustments']:
        print(f"- {adj}")
    for imp in recommendations['strategy_improvements']:
        print(f"- {imp}")

if __name__ == "__main__":
    main() 