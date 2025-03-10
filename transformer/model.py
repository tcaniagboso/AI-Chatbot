"""model.py: Defines the GPT language model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.layers import Block
from transformer.config import vocab_size, d_model, n_head, n_layer, block_size, device

class TransformerModel(nn.Module):
    """A simple decoder-only GPT-like model."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with a normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass for language modeling.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T).
            targets (torch.Tensor, optional): Target tensor for loss calculation.

        Returns:
            torch.Tensor: Logits or loss if targets are provided.
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
            loss = torch.nn.functional.cross_entropy(logits, targets)
            return logits, loss