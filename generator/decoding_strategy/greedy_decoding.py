from generator.decoding_strategy.decoding_strategy import DecodingStrategy
import torch

class GreedyDecoding(DecodingStrategy):
    """
    Greedy decoding strategy that selects the token with the highest probability.
    """
    def __init__(self, temperature: float = 0.9):
        """
        Initializes the Greedy decoding strategy.

        Parameters
        ----------
        temperature : float, optional
            The temperature value to scale logits before applying softmax (default is 0.9). 
            Lower values make the distribution more confident, higher values increase randomness.
        """
        self.temperature: float = temperature

    def select_next_token(self, logits: torch.Tensor) -> int:
        probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        return torch.argmax(probs).item()