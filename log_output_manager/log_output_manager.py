# log_output_manager.py
"""
Base interface for log output management.

Defines the contract for all log output implementations.
"""

class LogOutputManager:
    """
    Abstract base class for logging messages during training.

    Subclasses must implement the `log_message` method.
    """

    def log_message(self, message: str):
        """
        Log a message.

        Args:
            message (str): The message to log.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")