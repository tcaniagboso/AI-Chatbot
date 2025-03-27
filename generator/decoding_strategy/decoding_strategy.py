from abc import ABC, abstractmethod
import torch

class DecodingStrategy(ABC):
    """
    Base class for decoding strategies.
    Follows the Strategy pattern to encapsulate different token selection algorithms.
    """
    
    @abstractmethod
    def select_next_token(self, logits: torch.Tensor) -> int:
        """
        Selects the next token based on the strategy implementation.
        
        Args:
            logits (torch.Tensor): Logits for the next token.   
            
        Returns:
            int: The selected token ID.
        """
        pass