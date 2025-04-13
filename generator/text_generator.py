import torch
from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from generator.decoding_strategy.decoding_strategy import DecodingStrategy
from typing import List

class TextGenerator:
    """
    Handles text generation using a trained TransformerModel with:
    - Greedy decoding
    - Top-k sampling
    - Nucleus sampling (Top-p)
    - Repetition penalty to avoid loops
    """

    def __init__(self, model: TransformerModel, tokenizer: Tokenizer, decoding_strategy: DecodingStrategy):
        """
        Initializes the TextGenerator.

        Args:
            model (TransformerModel): Trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding/decoding text.
        """
        self.model: TransformerModel = model
        self.tokenizer: Tokenizer = tokenizer
        self.decoding_strategy: DecodingStrategy = decoding_strategy

    def generate_text(self, input_ids: List[int], max_length: int = 20) -> List[int]:
        """
        Generates text using the trained Transformer model.

        Args:
            input_ids (List[int]): Initial tokenized input.
            max_length (int): Maximum tokens to generate.
            
        Returns:
            List[int]: Generated sequence of token IDs.
        """
        if not input_ids:
            raise ValueError("Empty input provided to generate_text(). Check your tokenizer.")

        # Detect device from the model**
        device = next(self.model.parameters()).device  
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  #Move to the correct device

        for _ in range(max_length):
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
            next_token_id = self.decoding_strategy.select_next_token(next_token_logits)

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