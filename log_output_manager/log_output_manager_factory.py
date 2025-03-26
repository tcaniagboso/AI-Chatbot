from log_output_manager.log_output_manager import LogOutputManager
from log_output_manager.console_output_manager import ConsoleOutputManager
from log_output_manager.file_output_manager import FileOutputManager
from enum import Enum

class OutputManagerType(Enum):
    """
    Enum representing the available types of log output managers.

    Attributes:
        FILE (str): Logs to a file.
        CONSOLE (str): Logs to the terminal console.
    """
    FILE = "file"
    CONSOLE = "console"


class LogOutputManagerFactory:
    """
    Factory class responsible for creating instances of LogOutputManager subclasses.

    This factory enables switching between different logging outputs such as:
    - Console logging (ConsoleOutputManager)
    - File logging (FileOutputManager)

    Methods:
        create_log_output_manager(output_type, log_file): Creates and returns the selected log output manager.
    """

    @staticmethod
    def create_log_output_manager(output_type: OutputManagerType = OutputManagerType.CONSOLE, log_file: str = "logs/training_log.txt") -> LogOutputManager:
        """
        Creates a logger based on the desired output type.

        Args:
            output_type (OutputManagerType): Type of logging backend to use. Supported values: OutputManagerType.CONSOLE, OutputManagerType.FILE
            log_file (str): Path to the log file (used only if output_type is FILE).

        Returns:
            LogOutputManager: An instance of the selected log output manager.

        Raises:
            ValueError: If an unsupported output_type is provided.
        """
        if output_type == OutputManagerType.CONSOLE:
            return ConsoleOutputManager()
        elif output_type == OutputManagerType.FILE:
            return FileOutputManager(log_file)
        else:
            raise ValueError(f"Unknown LogOutputManagerType")
