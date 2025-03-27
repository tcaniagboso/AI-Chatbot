from generator.decoding_strategy.decoding_strategy import DecodingStrategy
import torch

class TopKDecoding(DecodingStrategy):
    """
    Top-K sampling strategy that selects from the K most probable tokens.
    """
    def __init__(self, k: int = 50, temperature: float = 0.9):
        """
        Initializes the Top-K decoding strategy.

        Parameters
        ----------
        k : int, optional
            The number of top logits to consider when sampling the next token (default is 50).
        temperature : float, optional
            The temperature value to scale logits before applying softmax (default is 0.9). 
            Lower values make the distribution more confident, higher values increase randomness.
        """
        self.k: int = k
        self.temperature: float = temperature
    
    def select_next_token(self, logits: torch.Tensor) -> int:
        probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.k)
        selected_index = torch.multinomial(topk_probs, 1).item()
        return topk_indices[selected_index].item()