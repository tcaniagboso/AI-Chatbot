import torch
from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from transformer.config import device
from typing import List


class TextGenerator:
    """
    Handles text generation using a trained TransformerModel with:
    - Greedy decoding
    - Top-k sampling
    - Nucleus sampling (Top-p)
    - Repetition penalty to avoid loops
    """

    def __init__(self, model: TransformerModel, tokenizer: Tokenizer):
        """
        Initializes the TextGenerator.

        Args:
            model (TransformerModel): Trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding/decoding text.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, input_ids: List[int], max_length: int = 20, decoding_strategy: str = 'top_k', temperature: float = 0.9) -> List[int]:
        """
        Generates text using the trained Transformer model.

        Args:
            input_ids (list[int]): Initial tokenized input.
            max_length (int): Maximum tokens to generate.
            decoding_strategy (str): Decoding method ('greedy', 'top_k', 'nucleus').
            temperature (float): Softmax temperature for randomness.

        Returns:
            list[int]: Generated sequence of token IDs.
        """
        if not input_ids:
            raise ValueError("Empty input provided to generate_text(). Check your tokenizer.")

        # Detect device from the model**
        device = next(self.model.parameters()).device  
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  #Move to the correct device

        for step in range(max_length):
            with torch.no_grad():
                logits, _ = self.model(input_tensor)  # Ensure all tensors are on the same device

            logits = logits.squeeze(0)  # Ensure correct shape

            if logits.size(0) == 0:
                raise RuntimeError("Model produced no output — check input format or model itself.")

            # Ensure the last token is on the same device
            next_token_logits = logits[-1, :].to(device)  

            # Apply repetition penalty BEFORE softmax
            next_token_logits = self.apply_repetition_penalty(next_token_logits, input_ids)

            # Select next token using the chosen decoding strategy
            next_token_id = self.select_next_token(next_token_logits, strategy=decoding_strategy, temperature=temperature)

            # Stop if we generate EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            input_ids.append(next_token_id)

            # Update input tensor on the correct device**
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  

        return input_ids


    def apply_repetition_penalty(self, logits: torch.Tensor, input_ids: List[int], penalty: float = 1.2) -> torch.Tensor:
        """
        Reduces probability of repeating the same tokens.

        Args:
            logits (torch.Tensor): Raw logits (pre-softmax).
            input_ids (List[int]): Previously generated tokens.
            penalty (float): Scaling factor (> 1.0 reduces repetition likelihood).

        Returns:
            torch.Tensor: Adjusted logits.
        """
        if input_ids:
            for token_id in set(input_ids):
                logits[token_id] /= penalty  # Apply penalty before softmax

        return logits

    def select_next_token(
        self, logits: torch.Tensor, strategy: str, temperature: float = 0.7, top_k: int = 50, p: float = 0.9
    ) -> int:
        """
        Selects the next token based on the decoding strategy.

        Args:
            logits (torch.Tensor): Logits for the next token.
            strategy (str): Decoding strategy ('greedy', 'top_k', 'nucleus').
            temperature (float): Temperature for scaling softmax.
            top_k (int): Number of top-k candidates.
            p (float): Nucleus sampling probability threshold.

        Returns:
            int: Chosen token ID.
        """
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

        if strategy == 'greedy':
            return torch.argmax(probs).item()

        elif strategy == 'top_k':
            topk_probs, topk_indices = torch.topk(probs, top_k)
            selected_index = torch.multinomial(topk_probs, 1).item()
            return topk_indices[selected_index].item()

        elif strategy == 'nucleus':
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            # Ensure at least one token is chosen
            cutoff = (cumulative_probs > p).nonzero(as_tuple=True)[0].min().item() + 1

            nucleus_probs = sorted_probs[:cutoff]
            nucleus_indices = sorted_indices[:cutoff]

            selected_index = torch.multinomial(nucleus_probs, 1).item()
            return nucleus_indices[selected_index].item()

        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")