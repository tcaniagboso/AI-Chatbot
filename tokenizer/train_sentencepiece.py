"""
Tokenizer Training Script

This script trains a SentencePiece tokenizer using the Wikitext-2 dataset.
It extracts raw text, saves it to a file, and then trains a Byte-Pair Encoding (BPE) tokenizer.

The trained tokenizer is saved as:
    - tokenizer/spm_model.model
    - tokenizer/spm_model.vocab

Example Usage:
    python train_sentencepiece.py
"""

import sentencepiece as spm
import os
from dataset_manager.dataset_manager import DatasetManager

def preprocess_text(text: str) -> str:
    """
    Preprocesses text before feeding it into the tokenizer training.
    
    - Removes excessive whitespace.
    - Converts to lowercase (optional, depending on dataset needs).
    - Ensures proper line formatting.

    Args:
        text (str): Raw text input.

    Returns:
        str: Cleaned text.
    """
    return " ".join(text.split()).strip()  # Simple whitespace cleanup

def train_tokenizer(output_prefix: str = 'tokenizer/spm_model', vocab_size: int = 32000, dataset_name: str = "wikitext"):
    """
    Trains a SentencePiece tokenizer using a specified dataset.

    Args:
        output_prefix (str): Prefix for the saved tokenizer files (model + vocab).
        vocab_size (int): Number of tokens in the final vocabulary.
        dataset_name (str): Name of the dataset to load (default: "wikitext").
    
    Steps:
        1. Load dataset (default: Wikitext-2).
        2. Extract training text and preprocess it.
        3. Write training text to 'training_text.txt'.
        4. Train tokenizer using BPE.
        5. Save tokenizer files in the 'tokenizer/' folder.

    Output Files:
        tokenizer/spm_model.model  - Trained SentencePiece model.
        tokenizer/spm_model.vocab  - Vocabulary file.
    """
    os.makedirs('tokenizer', exist_ok=True)

    dataset_manager = DatasetManager()
    dataset_manager.load_pretraining_dataset()

    # Extract and preprocess training text
    training_texts = dataset_manager.get_training_text()
    training_texts = [preprocess_text(text) for text in training_texts]  # Clean up the text

    # Save training text for SentencePiece
    training_text_path = 'training_text.txt'
    with open(training_text_path, 'w', encoding='utf-8') as f:
        for text in training_texts:
            f.write(text + '\n')

    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        input=training_text_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  # Covers most English words
        model_type='bpe',
        max_sentence_length=256,
        pad_id=0,  # Ensures pad token exists
        unk_id=1,  # Unknown token
        bos_id=2,  # Beginning of sentence token
        eos_id=3   # End of sentence token
    )

    print(f"[INFO] Tokenizer trained and saved as '{output_prefix}.model' and '{output_prefix}.vocab'")

if __name__ == '__main__':
    train_tokenizer()