# file_output_manager.py
"""
File-based implementation of LogOutputManager.

Logs messages to a specified text file.
"""

import os
from log_output_manager.log_output_manager import LogOutputManager

class FileOutputManager(LogOutputManager):
    """
    Logs messages to a file on disk.

    Attributes:
        log_file (str): Path to the log file.
    """

    def __init__(self, log_file: str = 'logs/training_log.txt'):
        """
        Initializes the file output manager.

        Ensures the directory for the log file exists.

        Args:
            log_file (str): Path to the log file.
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_message(self, message: str):
        """
        Writes a log message to the file.

        Args:
            message (str): The message to log.
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')