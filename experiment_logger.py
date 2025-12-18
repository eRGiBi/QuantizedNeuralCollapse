import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt


logger = logging.Logger(__name__)


class ExperimentLogger:
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = "exp",
        save_to_disk: bool = True,
    ):
        self.config = config or {}
        self.log_data = []
        self.save_to_disk = save_to_disk

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_{experiment_name}"

        if save_to_disk:
            self.exp_dir = Path(log_dir) / self.run_name
            self.exp_dir.mkdir(parents=True, exist_ok=True)

            self.config_path = self.exp_dir / "config.json"
            self.metrics_path = self.exp_dir / "metrics.csv"
            self.results_path = self.exp_dir / "results.json"

            # Save initial config
            self._save_json(self.config, self.config_path)
            logger.info(f"Experiment started at: {self.exp_dir}")

        else:
            self.exp_dir = None
            logger.info("Experiment logger running in dry mode (no files will be saved).")

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log a dictionary of metrics.

        Args:
            metrics: Dictionary of values (e.g., {'loss': 0.5, 'acc': 0.9})
            step: Optional epoch or iteration number.
        """
        log_entry = metrics.copy()
        log_entry["timestamp"] = datetime.now().isoformat()

        self.log_data.append(log_entry)

        if self.save_to_disk:
            self._append_to_csv(log_entry)

    def _append_to_csv(self, data: Dict[str, Any]):
        """Helper to append a row to CSV, handling new columns"""
        file_exists = self.metrics_path.exists()

        # If file doesn't exist, create it with current keys
        if not file_exists:
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                writer.writeheader()
                writer.writerow(data)
            return

        # check if keys match
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])

        new_keys = set(data.keys()) - set(headers)

        if new_keys:
            logger.warning(f"New metrics detected {new_keys}. Updating CSV headers.")
            df = pd.read_csv(self.metrics_path)
            for key in new_keys:
                df[key] = pd.NA

            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(self.metrics_path, index=False)
        else:

            with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(data)

    def save_results(self, results: Dict[str, Any]):
        """Save final results dict."""
        if self.save_to_disk:
            self._save_json(results, self.results_path)

    def update_config(self, params: Dict[str, Any]):
        """Update config and re-save."""
        self.config.update(params)
        if self.save_to_disk:
            self._save_json(self.config, self.config_path)

    @staticmethod
    def _save_json(data: Dict, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def __getitem__(self, key):
        return self.config.get(key)
       
            
    def save_model(self, model, filename: str = "model.pth"):
        """Save a PyTorch model to the experiment directory.
        """
        model_path = self.exp_dir / filename
        torch.save(model.state_dict(), model_path)
        
        print(f"Model saved to {model_path}")


    # def __setitem__(self, key: str, value: Any):
    #     """Set a configuration value."""
    #     self.config[key] = value
    #     if self.config_path:
    #         self._save_json(self.config)

    def __repr__(self) -> str:
        """String representation of the logger."""
        return f"ExperimentLogger(exp_dir={self.exp_dir}, config_keys={list(self.config.keys())})"
