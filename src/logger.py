import logging
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, Optional
import sys

class RetinalLogger:
    def __init__(self, log_dir: str = "../logs", experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name based on timestamp if not provided
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging configurations
        self.setup_loggers()
        
        # Dictionary to store metrics
        self.metrics: Dict[str, Any] = {}
        
    def setup_loggers(self):
        # Main logger
        self.logger = logging.getLogger(f"retinal_analysis_{self.experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # File handler for all logs
        fh = logging.FileHandler(
            self.log_dir / f"{self.experiment_name}_full.log"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        
        # File handler for errors only
        fh_errors = logging.FileHandler(
            self.log_dir / f"{self.experiment_name}_errors.log"
        )
        fh_errors.setLevel(logging.ERROR)
        fh_errors.setFormatter(file_formatter)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(fh_errors)
        self.logger.addHandler(ch)
        
    def log_model_parameters(self, params: Dict[str, Any]):
        """Log model parameters"""
        self.logger.info("Model Parameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")
        
        # Save parameters to JSON
        with open(self.log_dir / f"{self.experiment_name}_params.json", 'w') as f:
            json.dump(params, f, indent=4)
            
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {"value": value, "step": step} if step is not None else {"value": value}
        self.metrics[name].append(metric_entry)
        
        self.logger.info(f"Metric - {name}: {value}" + (f" (Step {step})" if step is not None else ""))
        
        # Save metrics to JSON
        with open(self.log_dir / f"{self.experiment_name}_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def log_training_step(self, epoch: int, batch: int, loss: float, metrics: Dict[str, float]):
        """Log training step information"""
        self.logger.info(
            f"Training - Epoch: {epoch}, Batch: {batch}, Loss: {loss:.4f}, "
            f"Metrics: {', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}"
        )
        
    def log_validation_results(self, epoch: int, metrics: Dict[str, float]):
        """Log validation results"""
        self.logger.info(
            f"Validation - Epoch: {epoch}, "
            f"Metrics: {', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}"
        )
        
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        
    def log_data_loading(self, source: str, num_samples: int):
        """Log data loading information"""
        self.logger.info(f"Loading data from {source}: {num_samples} samples")