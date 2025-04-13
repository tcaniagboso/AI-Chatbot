import torch
import pytest
from transformer.model import TransformerModel
from tokenizer.tokenizer import Tokenizer

@pytest.fixture(scope="module")
def tokenizer():
    """
    Fixture that returns a singleton Tokenizer instance.

    Ensures that the same tokenizer is reused across all model tests to avoid 
    redundant loading and to maintain consistency.

    Returns:
        Tokenizer: The singleton tokenizer instance.
    """
    return Tokenizer()

def test_model_output_shape(tokenizer):
    """
    Test that model returns logits with correct shape (B, T, vocab_size) when targets are not provided.

    Args:
        tokenizer (Tokenizer): Tokenizer used to get vocab size for input shape validation.
    """
    B, T = 2, 10
    dummy_input = torch.randint(0, tokenizer.get_vocab_size(), (B, T))
    model = TransformerModel()
    logits, _ = model(dummy_input)

    assert logits.shape == (B, T, tokenizer.get_vocab_size())

def test_model_with_targets(tokenizer):
    """
    Test that the model returns a valid scalar loss when input and target are provided.

    Args:
        tokenizer (Tokenizer): Used to match vocab range in test input and targets.
    """
    B, T = 2, 10
    x = torch.randint(0, tokenizer.get_vocab_size(), (B, T))
    y = torch.randint(0, tokenizer.get_vocab_size(), (B, T))
    model = TransformerModel()
    logits, loss = model(x, y)

    assert loss is not None
    assert isinstance(loss.item(), float)

def test_model_forward_pass_shape():
    """
    Test that the logits shape matches (B, T, vocab_size) and loss is None if no targets are passed.

    This test manually uses model.lm_head.out_features as the vocab size.
    """
    model = TransformerModel()
    B, T = 2, 8
    x = torch.randint(0, model.lm_head.out_features, (B, T))

    logits, loss = model(x)

    assert logits.shape == (B, T, model.lm_head.out_features)
    assert loss is None

def test_model_runs_without_targets():
    """
    Ensure that the model does not crash and produces valid output even without target labels.
    """
    model = TransformerModel()
    x = torch.randint(0, model.lm_head.out_features, (1, 5))
    logits, loss = model(x)

    assert logits.shape[1] == x.shape[1]
    assert loss is None