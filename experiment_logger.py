import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.Logger(__name__)


class ExperimentLogger:
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        create_experiment_dir: bool = True,
    ):
        """Initialize the ExperimentLogger.

        Args:
            config: Configuration dictionary to save
            log_dir: Base directory for logs
            experiment_name: Name of experiment (auto-generated if None)
            create_experiment_dir: If True, creates timestamped experiment directory
        """
        self.config = config or {}
        self.log_dir = Path(log_dir)
        self.log_data = []
        self.log_fieldnames = []
        
        if create_experiment_dir:
            exp_name = experiment_name or self.config.get("exp_name", "exp")
            self.exp_dir = self.log_dir / f"{config["now"]}_{exp_name}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)

            self.config_path = self.exp_dir / "config.json"
            self.log_file = self.exp_dir / "logs.txt"
            self.metrics = self.exp_dir / "training_metrics.csv"
            self.results_path = self.exp_dir / "results.json"

            self._save_config()
            print(f"Experiment directory created at: {self.exp_dir}")
            
        else:
            self.exp_dir = self.log_dir
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.config_path = None
            self.log_file = None
            self.results_path = None

        # self.exp_dir = os.path.join(log_dir, f"{config["now"]}_{config.get('experiment_name', 'exp')}")
        
        # os.makedirs(self.exp_dir, exist_ok=True)
        
        # with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
        #     json.dump(self.config, f, indent=4)
            
        # self.log_file = os.path.join(self.exp_dir, 'logs.csv')
        # self.log_fieldnames = []
        

    def _save_config(self):
        """Save configuration to JSON file."""
        if self.config_path:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
                

    def load_config(self, file_path: Union[str, Path]):
        """Load configuration from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        return self.config
    

    def update_config(self, params_dict: Dict[str, Any], save: bool = True):
        """Update configuration with new parameters."""
        self.config.update(params_dict)
        if save and self.config_path:
            self._save_config()

    # def log(self, data):
    #     if not self.log_fieldnames:
    #         self.log_fieldnames = list(data.keys())
    #         with open(self.log_file, 'w', newline='') as f:
    #             writer = csv.DictWriter(f, fieldnames=self.log_fieldnames)
    #             writer.writeheader()
        
    #     with open(self.log_file, 'a', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=self.log_fieldnames)
    #         writer.writerow(data)
            
    def log(self, **kwargs):
        """Log metrics/data as key-value pairs."""
        log_entry = kwargs.copy()

        if "timestamp" not in log_entry:
            log_entry["timestamp"] = datetime.now().isoformat()

        self.log_data.append(log_entry)

        if self.metrics:
            self._append_to_csv(log_entry)
            
    def log_training_metrics(self, kwargs):
        """Log metrics/data as key-value pairs."""
        log_entry = kwargs.copy()

        if "timestamp" not in log_entry:
            log_entry["timestamp"] = datetime.now().isoformat()

        self.log_data.append(log_entry)

        self._append_to_metrics(log_entry)
            
    def _append_to_metrics(self, data: Dict[str, Any]):
        """Append a single row to the CSV log file."""
        file_exists = self.metrics.exists()

        if not file_exists or not self.log_fieldnames:
            self.log_fieldnames = list(data.keys())

        with open(self.metrics, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)
            
            
    def save_results(self, results: Dict[str, Any], file_path: Optional[Union[str, Path]] = None):
        """Save final results to a JSON file.
        """
        save_path = Path(file_path) if file_path else self.results_path

        if save_path is None:
            save_path = self.exp_dir / "results.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {save_path}")        
       
            
    def save_model(self, model, filename: str = "model.pth"):
        """Save a PyTorch model to the experiment directory.
        """
        model_path = self.exp_dir / filename
        torch.save(model.state_dict(), model_path)
        
        print(f"Model saved to {model_path}")


    def get_log_data(self):
        """Return all logged data."""
        return self.log_data

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value."""
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
        if self.config_path:
            self._save_config()

    def __repr__(self) -> str:
        """String representation of the logger."""
        return f"ExperimentLogger(exp_dir={self.exp_dir}, config_keys={list(self.config.keys())})"
