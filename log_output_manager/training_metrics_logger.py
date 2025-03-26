import csv
import os
import time
from typing import Dict, Any
from log_output_manager.observer import Observer
from log_output_manager.event_enums import EventType, MetricKey

class TrainingMetricsLogger(Observer):
    """
    Handles structured metric logging (loss curves, timing, etc.) to CSV.
    Separate from human-readable logs.
    """

    def __init__(self, 
                 training_filepath: str ='logs/training_metrics.csv', 
                 validation_filepath: str = 'logs/validation_metrics.csv', 
                 perplexity_filepath: str = 'logs/perplexity_metrics.csv'):
        """
        Initializes the TrainingMetricsLogger.

        Creates necessary directories and CSV files (if they do not already exist) for:
        - Training loss logs
        - Validation loss logs
        - Perplexity logs

        Args:
            training_filepath (str): Path to CSV file for training loss logs.
            validation_filepath (str): Path to CSV file for validation loss logs.
            perplexity_filepath (str): Path to CSV file for perplexity metrics.
        """
        
        os.makedirs(os.path.dirname(training_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(validation_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(perplexity_filepath), exist_ok=True)

        self.last_timestamp: float = None
        self.training_filepath: str = training_filepath
        self.validation_filepath: str = validation_filepath
        self.perplexity_filepath: str = perplexity_filepath

        if not os.path.exists(training_filepath):
            with open(training_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step_or_phase', 'loss', 'time_elapsed'])
        
        if not os.path.exists(validation_filepath):
            with open(validation_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'Validation Loss'])
        
        if not os.path.exists(self.perplexity_filepath):
            with open(self.perplexity_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model Type', 'Perplexity'])
        

    def update(self, event_type: EventType, data: Dict[MetricKey, Any]) -> None:
        """
        Receives notifications from the subject (PerformanceEvaluator) and routes them 
        to the appropriate logging function.

        Args:
            event_type (EventType): Type of event (e.g., 'training_loss', 'validation_loss', 'final_perplexity').
            data (Dict[MetricKey, Any]): Dictionary containing data relevant to the event.
        """
        if event_type == EventType.TRAINING_LOSS:
            self.log_training_loss(data[MetricKey.EPOCH], data[MetricKey.STEP], data[MetricKey.LOSS])
        elif event_type == EventType.VALIDATION_LOSS:
            self.log_validation_loss(data[MetricKey.EPOCH], data[MetricKey.LOSS])
        elif event_type == EventType.FINAL_SUMMARY:
            self.log_final_summary(data[MetricKey.TOTAL_TIME])
        elif event_type == EventType.FINAL_PERPLEXITY:
            self.log_perplexity(data[MetricKey.MODEL], data[MetricKey.PPL])

    def start_timer(self) -> None:
        """
        Starts the timer before the first batch/step.
        Call this at the start of training and after loading checkpoints.
        """
        self.last_timestamp = time.time()

    def log_training_loss(self, epoch: int, step: int, loss: float) -> None:
        """
        Logs training loss along with the time taken since the last log.

        Args:
            epoch (int): Current epoch.
            step (int): Current step within the epoch.
            loss (float): Training loss value.
        """
        if self.last_timestamp is None:
            elapsed = 0.0  # First log has no reference point
        else:
            elapsed = time.time() - self.last_timestamp

        with open(self.training_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, elapsed])

        self.last_timestamp = time.time()

    def log_validation_loss(self, epoch: int, val_loss: float) -> None:
        """
        Logs validation loss (only once per epoch).

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss after this epoch.
        """
        with open(self.validation_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_loss])

    def log_final_summary(self, total_training_time: float) -> None:
        """
        Logs total training time at the end of the existing metrics file.

        Args:
            total_training_time (float): Total training time in seconds.
        """
        with open(self.training_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])  # Blank line
            writer.writerow(['Training Completed', '', '', ''])
            writer.writerow(['Total Training Time (seconds)', total_training_time, '', ''])

    def log_perplexity(self, model_type: str, perplexity: float) -> None:
        """
        Logs perplexity result for a given model evaluation.

        Args:
            model_type (str): Type of model evaluated (e.g., "Custom Model").
            perplexity (float): Computed perplexity score.
        """
        with open(self.perplexity_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_type, perplexity])