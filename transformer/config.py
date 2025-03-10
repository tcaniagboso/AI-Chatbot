import torch

"""config.py: Defines hyperparameters for the GPT model."""

# Hyperparameters
batch_size = 8  # Number of independent sequences processed in parallel
block_size = 256  # Maximum context length
learning_rate = 3e-4
d_model = 512  # Embedding size (changed from 384)
n_head = 8  # Number of attention heads (changed from 6)
n_layer = 6  # Number of Transformer blocks
dropout = 0.2
vocab_size = 32000  # Adjust based on tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"