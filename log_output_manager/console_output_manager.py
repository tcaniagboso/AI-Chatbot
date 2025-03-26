# console_output_manager.py
"""
Console-based implementation of LogOutputManager.

Logs messages directly to the console (stdout).
"""

from log_output_manager.log_output_manager import LogOutputManager

class ConsoleOutputManager(LogOutputManager):
    """
    Logs messages directly to the console.
    """

    def log_message(self, message: str) -> None:
        """
        Prints a log message to the console.

        Args:
            message (str): The message to log.
        """
        print(message)