"""
Unit tests for the TextGenerator component.

These tests verify:
- Text generation works with all decoding strategies (Greedy, Top-K, Nucleus).
- Repetition penalty reduces probability for repeated tokens.
- Integration between TransformerModel, Tokenizer, and DecodingStrategy.
"""

import pytest
import torch
from generator.text_generator import TextGenerator
from generator.decoding_strategy.decoding_strategy_factory import DecodingStrategyFactory, DecodingStrategyType
from transformer.model import TransformerModel
from tokenizer.tokenizer import Tokenizer

@pytest.fixture(scope="module")
def tokenizer():
    """
    Fixture providing a singleton Tokenizer instance.

    Returns:
        Tokenizer: Shared tokenizer used in generation tests.
    """
    return Tokenizer()

@pytest.fixture(scope="module")
def model():
    """
    Fixture providing a fresh TransformerModel instance.

    Returns:
        TransformerModel: Lightweight model for test inference.
    """
    return TransformerModel()

def test_text_generation_runs(tokenizer, model):
    """
    Test that text generation using greedy decoding produces an output sequence
    at least as long as the input prompt.

    Args:
        tokenizer (Tokenizer): Shared tokenizer fixture.
        model (TransformerModel): Shared model fixture.
    """
    decoding_strategy = DecodingStrategyFactory.create_decoding_strategy(DecodingStrategyType.GREEDY)
    generator = TextGenerator(model, tokenizer, decoding_strategy)
    prompt = tokenizer.encode("The quick brown fox")
    output = generator.generate_text(prompt, max_length=5)

    assert len(output) >= len(prompt)

def test_repetition_penalty_changes_logits(tokenizer, model):
    """
    Test that applying the repetition penalty decreases logits for repeated tokens.

    Args:
        tokenizer (Tokenizer): Tokenizer used to get vocab size.
        model (TransformerModel): Required to construct TextGenerator.
    """
    decoding_strategy = DecodingStrategyFactory.create_decoding_strategy(DecodingStrategyType.GREEDY)
    generator = TextGenerator(model, tokenizer, decoding_strategy)

    logits = torch.ones(tokenizer.get_vocab_size())
    repeated_ids = [1, 1, 2]

    modified_logits = generator.apply_repetition_penalty(logits.clone(), repeated_ids)

    assert modified_logits[1] < logits[1], "Repetition penalty not applied correctly to token 1"
    assert modified_logits[2] < logits[2], "Repetition penalty not applied correctly to token 2"

def test_greedy_decoding_runs(tokenizer, model):
    """
    Ensure text is generated correctly using greedy decoding strategy.

    Args:
        tokenizer (Tokenizer): Tokenizer fixture.
        model (TransformerModel): Model fixture.
    """
    strategy = DecodingStrategyFactory.create_decoding_strategy(DecodingStrategyType.GREEDY)
    generator = TextGenerator(model, tokenizer, strategy)
    prompt = tokenizer.encode("The sky is")
    output = generator.generate_text(prompt, max_length=5)

    assert len(output) >= len(prompt)

def test_top_k_decoding_runs(tokenizer, model):
    """
    Ensure text is generated correctly using Top-K decoding.

    Args:
        tokenizer (Tokenizer): Tokenizer fixture.
        model (TransformerModel): Model fixture.
    """
    strategy = DecodingStrategyFactory.create_decoding_strategy(DecodingStrategyType.TOP_K)
    generator = TextGenerator(model, tokenizer, strategy)
    prompt = tokenizer.encode("The sky is")
    output = generator.generate_text(prompt, max_length=5)

    assert len(output) >= len(prompt)

def test_nucleus_decoding_runs(tokenizer, model):
    """
    Ensure text is generated correctly using Nucleus (Top-p) decoding.

    Args:
        tokenizer (Tokenizer): Tokenizer fixture.
        model (TransformerModel): Model fixture.
    """
    strategy = DecodingStrategyFactory.create_decoding_strategy(DecodingStrategyType.NUCLEUS)
    generator = TextGenerator(model, tokenizer, strategy)
    prompt = tokenizer.encode("The sky is")
    output = generator.generate_text(prompt, max_length=5)

    assert len(output) >= len(prompt)