from generator.decoding_strategy.decoding_strategy import DecodingStrategy
import torch

class NucleusDecoding(DecodingStrategy):
    """
    Nucleus (Top-p) sampling strategy that selects from the smallest set of tokens
    whose cumulative probability exceeds threshold p.
    """

    def __init__(self, p: float = 0.9, temperature: float = 0.9):
        """
        Initializes the Nucleus (Top-p) decoding strategy.

        Args:
            p (float, optional): The cumulative probability threshold (0 < p <= 1). 
                Only the smallest set of tokens whose probabilities sum to `p` are considered.
                Defaults to 0.9.
            temperature (float, optional): Scaling factor for softmax distribution. 
                Higher values increase randomness. Defaults to 0.9.
        """
        self.p: float = p
        self.temperature: float = temperature

    def select_next_token(self, logits: torch.Tensor) -> int:
        probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Ensure at least one token is chosen
        cutoff = (cumulative_probs > self.p).nonzero(as_tuple=True)[0]
        if len(cutoff) > 0:
            cutoff = cutoff.min().item() + 1
        else:
            cutoff = 1  # Fallback to the most probable token if no token meets threshold

        nucleus_probs = sorted_probs[:cutoff]
        nucleus_indices = sorted_indices[:cutoff]

        selected_index = torch.multinomial(nucleus_probs, 1).item()
        return nucleus_indices[selected_index].item()