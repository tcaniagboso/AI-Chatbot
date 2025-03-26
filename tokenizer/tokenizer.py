import sentencepiece as spm
from typing import List
from singleton.singleton import Singleton

class Tokenizer(Singleton):
    """
    Handles tokenization and detokenization using SentencePiece.

    This class is responsible for:
    - Encoding text into token IDs (int)
    - Decoding token IDs back into text
    - Encoding text as subword pieces (strings)
    - Providing special token IDs (EOS, PAD)
    """

    def _init_singleton(self, model_path: str = 'tokenizer/spm_model.model'):
        """
        Initializes the Tokenizer.

        Args:
            model_path (str): Path to the trained SentencePiece model file.
        """
        self.sp: spm.SentencePieceProcessor = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size: int = self.sp.vocab_size()

    def encode(self, text: str) -> List[int]:
        """
        Converts text into token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[int]: A list of token IDs representing the input text.
        """
        #print(f"[Tokenizer] Encoding text: {text}")
        tokens = self.sp.encode(text, out_type=int)
        #print(f"[Tokenizer] Tokens: {tokens}")
        return tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Converts token IDs back to text.

        Args:
            token_ids (List[int]): A list of token IDs.

        Returns:
            str: The reconstructed text from the token IDs.
        """
        return self.sp.decode(token_ids)
        
    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Converts text into subword tokens.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of tokenized subwords.
        """
        pieces = self.sp.encode(text, out_type=str)
        #print(f"[Tokenizer] Subword Pieces: {pieces}")  # Debug log
        return pieces
        
    @property
    def eos_token_id(self) -> int:
        """Returns the ID of the End-of-Sequence token, ensuring it's valid."""
        eos_id = self.sp.eos_id()
        return eos_id if eos_id >= 0 else None

    @property
    def pad_token_id(self) -> int:
        """Returns the ID of the padding token, ensuring it's valid."""
        pad_id = self.sp.pad_id()
        return pad_id if pad_id >= 0 else None
        
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the SentencePiece model."""
        return self.sp.vocab_size()