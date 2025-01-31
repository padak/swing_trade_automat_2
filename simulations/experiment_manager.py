"""
Experiment manager for handling trading simulations.
Loads configurations and manages experiment results.
"""

import os
import json
import importlib.util
from datetime import datetime
from typing import Dict, Any

class ExperimentManager:
    def __init__(self, experiment_path: str):
        """
        Initialize experiment manager.
        
        Args:
            experiment_path: Path to experiment directory (e.g., 'simulations/v1')
        """
        self.experiment_path = experiment_path
        self.config = self._load_config()
        self.results_dir = os.path.join(experiment_path, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_config(self):
        """Load experiment configuration from the specified path."""
        config_path = os.path.join(self.experiment_path, 'config.py')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.py found in {self.experiment_path}")

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.ExperimentConfig

    def get_parameters(self) -> Dict[str, Any]:
        """Get experiment parameters."""
        return self.config.get_params()

    def save_results(self, results: Dict[str, Any]):
        """
        Save experiment results.
        
        Args:
            results: Dictionary containing experiment results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'results_{timestamp}.json')
        
        # Add experiment metadata
        results['experiment'] = {
            'name': self.config.NAME,
            'description': self.config.DESCRIPTION,
            'version': self.config.VERSION,
            'timestamp': timestamp,
            'parameters': self.get_parameters()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")

    def load_results(self, results_file: str) -> Dict[str, Any]:
        """
        Load results from a specific file.
        
        Args:
            results_file: Name of the results file to load
        
        Returns:
            Dictionary containing the results
        """
        file_path = os.path.join(self.results_dir, results_file)
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_all_results(self) -> list:
        """Get all results for this experiment."""
        results = []
        for file in os.listdir(self.results_dir):
            if file.endswith('.json'):
                results.append(self.load_results(file))
        return results 