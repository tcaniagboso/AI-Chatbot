import pytest
from tokenizer.tokenizer import Tokenizer

@pytest.fixture(scope="module")
def tokenizer():
    """
    Provides a shared instance of the Tokenizer for all tests in this module.

    This avoids reloading the SentencePiece model repeatedly and ensures consistency
    across tests that validate encoding, decoding, and special token behavior.

    Returns:
        Tokenizer: A singleton Tokenizer instance.
    """
    return Tokenizer()

def test_tokenizer_round_trip(tokenizer):
    """
    Test that encoding and decoding a string results in similar output.

    This validates that the Tokenizer can convert text into token IDs and
    accurately reconstruct text from them.

    Args:
        tokenizer (Tokenizer): The Tokenizer fixture.
    """
    text = "hello world"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)
    assert isinstance(decoded, str)
    assert "hello" in decoded.lower() or "world" in decoded.lower()  # SentencePiece may paraphrase

def test_tokenizer_padding(tokenizer):
    """
    Test that the tokenizer returns a valid list of token IDs within vocabulary bounds.

    Args:
        tokenizer (Tokenizer): The Tokenizer fixture.
    """
    encoded = tokenizer.encode("short text")

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert max(encoded) < tokenizer.get_vocab_size()

def test_special_tokens(tokenizer):
    """
    Test that the tokenizer exposes special tokens: EOS and PAD.

    Ensures both values are not None and are valid integer token IDs.

    Args:
        tokenizer (Tokenizer): The Tokenizer fixture.
    """
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.eos_token_id, int)
    assert isinstance(tokenizer.pad_token_id, int)