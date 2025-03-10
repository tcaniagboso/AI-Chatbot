"""layers.py: Defines the transformer layers for the GPT model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.config import d_model, block_size, dropout

class Head(nn.Module):
    """A single head of self-attention.

    Args:
        head_size (int): Dimensionality of the attention head.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for self-attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: Output of self-attention layer.
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
    """Multi-head self-attention layer.

    Args:
        num_heads (int): Number of attention heads.
        head_size (int): Dimensionality of each head.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Multi-head attention output.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """Feedforward network with ReLU activation."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward pass for the feedforward layer."""
        return self.net(x)

class Block(nn.Module):
    """A single transformer block.

    Args:
        n_embd (int): Embedding size.
        n_head (int): Number of attention heads.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Forward pass for the transformer block."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x