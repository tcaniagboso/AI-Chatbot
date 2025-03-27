from view.iview import IView

class ConsoleView(IView):
    """
    Console-based implementation of the IView interface.
    Handles input/output through the command line interface.
    """
    
    def get_user_input(self) -> str:
        """
        Gets input from the user via the console.
        
        Returns:
            str: The user's input string.
        """
        return input(">> ")
    
    def display_output(self, output: str) -> None:
        """
        Displays output to the user via the console.
        
        Args:
            output (str): The output string to display.
        """
        print(f'<< {output}')
        print()
