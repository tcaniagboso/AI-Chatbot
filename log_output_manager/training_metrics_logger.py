import csv
import os
import time

class TrainingMetricsLogger:
    """
    Handles structured metric logging (loss curves, timing, etc.) to CSV.
    Separate from human-readable logs.
    """

    def __init__(self, training_filepath='logs/training_metrics.csv', validation_filepath = 'logs/validation_metrics.csv'):
        os.makedirs(os.path.dirname(training_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(validation_filepath), exist_ok=True)
        self.training_filepath = training_filepath
        self.validation_filepath = validation_filepath

        if not os.path.exists(training_filepath):
            with open(training_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step_or_phase', 'loss', 'time_elapsed'])
        
        if not os.path.exists(validation_filepath):
            with open(validation_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'Validation Loss'])

        self.last_timestamp = None

    def start_timer(self):
        """
        Starts the timer before the first batch/step.
        Call this at the start of training and after loading checkpoints.
        """
        self.last_timestamp = time.time()

    def log_training_loss(self, epoch, step, loss):
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

    def log_validation_loss(self, epoch, val_loss):
        """
        Logs validation loss (only once per epoch).

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss after this epoch.
        """
        with open(self.validation_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_loss])

    def log_final_summary(self, total_training_time):
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