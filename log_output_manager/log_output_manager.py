# log_output_manager.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from log_output_manager.observer import Observer
from log_output_manager.event_enums import EventType, MetricKey

"""
Base interface for log output management.

Defines the contract for all log output implementations.
"""

class LogOutputManager(Observer, ABC):
    """
    Abstract base class for logging messages during training.

    Subclasses must implement the `log_message` method.
    """

    @abstractmethod
    def log_message(self, message: str) -> None:
        """
        Log a message.

        Args:
            message (str): The message to log.
        """
        pass

    def update(self, event_type: EventType, data: Dict[MetricKey, Any]) -> None:
        """
        Handles observer update. Delegates to log_message using formatted string.

        Args:
            event_type (EventType): Type of event ('training_loss', 'val_loss', etc.)
            data (Dict[MetricKey, Any]): Data associated with the event.
        """
        if event_type == EventType.TRAINING_LOSS:
            message = f"[TRAIN] Epoch {data[MetricKey.EPOCH]} | Step {data[MetricKey.STEP]} | Loss: {data[MetricKey.LOSS]:.4f}"
        elif event_type == EventType.VALIDATION_LOSS:
            message = f"[VALIDATION] Epoch {data[MetricKey.EPOCH]} | Validation Loss: {data[MetricKey.LOSS]:.4f}"
        elif event_type == EventType.FINAL_SUMMARY:
            message = f"[SUMMARY] Training completed in {data[MetricKey.TOTAL_TIME]:.2f} seconds"
        elif event_type == EventType.FINAL_PERPLEXITY:
            message = f"[PERPLEXITY] Final Perplexity on Test Set ({data[MetricKey.MODEL]}): {data[MetricKey.PPL]:.4f}"
        elif event_type == EventType.EVALUATING_PERPLEXITY:
            message = f"[EVALUATION] Evaluating Perplexity on {data[MetricKey.MODEL]}..."
        else:
            # Generic fallback
            message = f"[{EventType.UNKNOWN.value}] " + " | ".join(f"{k.value}: {v}" for k, v in data.items())

        self.log_message(message)
