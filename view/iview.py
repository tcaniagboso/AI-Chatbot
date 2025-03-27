from abc import ABC, abstractmethod

class IView(ABC):
    """
    Abstract interface for all views in the application.
    Defines the essential methods that any view must implement.
    """
    
    @abstractmethod
    def get_user_input(self) -> str:
        """
        Gets input from the user.
        
        Returns:
            str: The user's input.
        """
        pass
    
    @abstractmethod
    def display_output(self, output: str) -> None:
        """
        Displays output to the user.
        
        Args:
            output (str): The output to display.
        """
        pass
