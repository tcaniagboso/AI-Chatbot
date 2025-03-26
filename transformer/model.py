"""model.py: Defines the GPT language model."""

import torch
import torch.nn as nn
from transformer.layers import Block
from transformer.config import vocab_size, d_model, n_head, n_layer, block_size
from typing import Tuple, Optional

class TransformerModel(nn.Module):
    """
    A simple decoder-only GPT-like model.

    Attributes:
        token_embedding_table (nn.Embedding): Embeds input token IDs into dense vectors.
        position_embedding_table (nn.Embedding): Encodes positional information.
        blocks (nn.Sequential): Stack of Transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Output layer that maps to vocabulary size.
    """

    def __init__(self):
        """
        Initializes the TransformerModel with token and position embeddings,
        transformer blocks, normalization, and output head.
        """
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table: nn.Embedding = nn.Embedding(block_size, d_model)
        self.blocks: nn.Sequential = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f: nn.LayerNorm = nn.LayerNorm(d_model)
        self.lm_head: nn.Linear = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights using a normal distribution.

        Args:
            module (nn.Module): Module whose weights will be initialized.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass for language modeling.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T) containing token indices.
            targets (torch.Tensor, optional): Ground truth target tensor of shape (B, T).
        
        Returns:
            Tuple: A tuple containing:
                - logits (torch.Tensor): Output predictions of shape (B, T, vocab_size).
                - loss (torch.Tensor | None): Cross-entropy loss if targets are provided, else None.
        """
        device = idx.device  # Get the correct device dynamically
        B, T = idx.shape

        # Ensure `torch.arange(T)` is on the same device as `idx`
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Move to same device

        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)  # Transformer layers
        x = self.ln_f(x)  # Final normalization
        logits = self.lm_head(x)  # Output logits

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)
            return logits, loss