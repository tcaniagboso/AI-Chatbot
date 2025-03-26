"""layers.py: Defines the transformer layers for the GPT model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.config import d_model, block_size, dropout

class Head(nn.Module):
    """
    A single head of self-attention.

    Args:
        head_size (int): Dimensionality of the attention head.

    Attributes:
        key (nn.Linear): Linear layer to project input to key.
        query (nn.Linear): Linear layer to project input to query.
        value (nn.Linear): Linear layer to project input to value.
        tril (torch.Tensor): Lower triangular matrix for causal masking.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, head_size: int):
        super().__init__()
        self.key: nn.Linear = nn.Linear(d_model, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(d_model, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Args:
        num_heads (int): Number of attention heads.
        head_size (int): Dimensionality of each head.
    
    Attributes:
        heads (nn.ModuleList): List of attention heads.
        proj (nn.Linear): Output projection layer.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj: nn.Linear = nn.Linear(head_size * num_heads, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Multi-head attention output.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """
    Feedforward network with ReLU activation.

    Args:
        n_embd (int): Embedding size.

    Attributes:
        net (nn.Sequential): Feedforward network pipeline.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feedforward layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)

class Block(nn.Module):
    """
    A single transformer block.

    Args:
        n_embd (int): Embedding size.
        n_head (int): Number of attention heads.

    Attributes:
        sa (MultiHeadAttention): Multi-head self-attention module.
        ffwd (FeedForward): Feedforward network module.
        ln1 (nn.LayerNorm): Layer norm before self-attention.
        ln2 (nn.LayerNorm): Layer norm before feedforward.
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size: int = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size)
        self.ffwd: FeedForward = FeedForward(n_embd)
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x