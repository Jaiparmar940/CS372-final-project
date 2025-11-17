"""
Logging utilities for training metrics.
"""

import csv
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Logger:
    """
    Simple logger for training metrics.
    
    Supports CSV and JSON logging.
    """
    
    def __init__(
        self,
        log_dir: str = "results",
        log_file: str = "training.log",
        use_csv: bool = True,
        use_json: bool = False,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            log_file: Name of log file
            use_csv: Whether to log to CSV
            use_json: Whether to log to JSON
        """
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.use_csv = use_csv
        self.use_json = use_json
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV setup
        if self.use_csv:
            self.csv_path = self.log_dir / f"{log_file}.csv"
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = None
            self.csv_keys = None
        
        # JSON setup
        if self.use_json:
            self.json_path = self.log_dir / f"{log_file}.json"
            self.json_entries = []
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        # CSV logging
        if self.use_csv:
            if self.csv_writer is None:
                # Initialize CSV writer with keys from first entry
                self.csv_keys = list(metrics.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_keys)
                self.csv_writer.writeheader()
            
            # Write row
            row = {k: metrics.get(k, '') for k in self.csv_keys}
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        
        # JSON logging
        if self.use_json:
            self.json_entries.append(metrics.copy())
            with open(self.json_path, 'w') as f:
                json.dump(self.json_entries, f, indent=2)
    
    def close(self):
        """Close log files."""
        if self.use_csv and self.csv_file:
            self.csv_file.close()
        
        if self.use_json:
            with open(self.json_path, 'w') as f:
                json.dump(self.json_entries, f, indent=2)

