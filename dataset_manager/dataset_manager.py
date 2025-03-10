from datasets import load_dataset
from typing import List, Dict, Optional, Union
import random

from singleton.singleton import Singleton


class DatasetManager:
    """
    Manages loading and access to different datasets used for training and fine-tuning.

    Supports:
    - Pretraining: Next-word prediction (e.g., OpenWebText, Wikitext-2)
    - Classification: Sentiment analysis (IMDb)
    - QA: Instruction tuning (Alpaca)
    """

    def __init__(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1", split_ratios: tuple = (0.8, 0.1, 0.1)):
        """
        Initializes the dataset manager.

        Args:
            dataset_name (str): Name of the dataset to load (default: "wikitext").
            subset (str): Subset of the dataset (default: "wikitext-2-raw-v1").
            split_ratios (tuple): Ratios for splitting dataset if needed (train, validation, test).
        """
        self.dataset: Optional[Dict[str, object]] = None
        self.dataset_name = dataset_name
        self.subset = subset
        self.split_ratios = split_ratios  # Used for OpenWebText

    def load_pretraining_dataset(self, dataset_name: str = None):
        """
        Loads a dataset for next-word prediction (pretraining task).

        Args:
            dataset_name (str, optional): Name of the dataset to load. Defaults to `self.dataset_name`.
        """
        dataset_name = dataset_name or self.dataset_name

        try:
            if dataset_name == "openwebtext":
                full_dataset = load_dataset("openwebtext")["train"]  # OpenWebText has only 'train' split
                self.dataset = self._split_openwebtext(full_dataset)
            elif dataset_name == "wikitext":
                self.dataset = load_dataset("wikitext", self.subset)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            print(f"Successfully loaded {dataset_name} dataset.")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            self.dataset = None

    def load_classification_dataset(self):
        """Loads the IMDb dataset for sentiment classification."""
        try:
            self.dataset = load_dataset("imdb")
            print("IMDb dataset loaded.")
        except Exception as e:
            print(f"Error loading IMDb dataset: {e}")

    def load_qa_dataset(self):
        """Loads the Alpaca dataset for instruction tuning & question answering."""
        try:
            self.dataset = load_dataset("tatsu-lab/alpaca")
            print("Alpaca dataset loaded.")
        except Exception as e:
            print(f"Error loading Alpaca dataset: {e}")

    def _split_openwebtext(self, dataset):
        """
        Splits OpenWebText dataset into train, validation, and test sets.

        Args:
            dataset: The full OpenWebText dataset (single split).

        Returns:
            Dict[str, List[str]]: Splitted dataset dictionary.
        """
        if dataset is None:
            print("OpenWebText dataset is empty. Aborting split.")
            return {"train": [], "validation": [], "test": []}

        texts = [sample["text"] for sample in dataset if sample["text"].strip()]
        
        if not texts:
            print("No valid text samples found in OpenWebText dataset.")
            return {"train": [], "validation": [], "test": []}

        random.shuffle(texts)  # Shuffle data before splitting

        total = len(texts)
        train_end = int(total * self.split_ratios[0])
        val_end = train_end + int(total * self.split_ratios[1])

        return {
            "train": texts[:train_end],
            "validation": texts[train_end:val_end],
            "test": texts[val_end:],
        }

    def _get_text_split(self, split: str) -> List[str]:
        """
        Retrieves non-blank text samples from a specified dataset split.

        Args:
            split (str): Dataset split ("train", "validation", or "test").

        Returns:
            List[str]: List of cleaned text lines.
        """
        if self.dataset is None:
            print("No dataset loaded. Please call `load_pretraining_dataset()` first.")
            return []

        if split not in self.dataset:
            print(f"Dataset does not contain split: {split}")
            return []

        # If OpenWebText, return list of texts directly
        if isinstance(self.dataset[split], list):  # OpenWebText case
            return self.dataset[split]

        # If Wikitext, extract "text" field (this was missing before!)
        return [sample["text"] for sample in self.dataset[split] if isinstance(sample, dict) and "text" in sample]


    def get_training_text(self) -> List[str]:
        """Retrieves training split text samples."""
        return self._get_text_split("train")

    def get_validation_text(self) -> List[str]:
        """Retrieves validation split text samples."""
        return self._get_text_split("validation")

    def get_test_text(self) -> List[str]:
        """Retrieves test split text samples."""
        return self._get_text_split("test")


class SingletonDatasetManager(DatasetManager, Singleton):
    """
    Singleton wrapper for DatasetManager to ensure only one instance exists.
    """

    def _init_singleton(self):
        """Called once to initialize the singleton instance."""
        super().__init__()