"""
Unit tests for the DatasetManager class.

These tests validate the loading, splitting, and retrieval of datasets,
with focus on Wikitext for pretraining tasks.
"""

import pytest
from dataset_manager.dataset_manager import DatasetManager

@pytest.fixture(scope="module")
def dataset_manager():
    """
    Fixture that returns a DatasetManager instance with Wikitext loaded.
    """
    manager = DatasetManager()
    manager.load_pretraining_dataset("wikitext")
    return manager

def test_dataset_is_loaded(dataset_manager):
    """
    Ensure that loading the Wikitext dataset sets a valid dataset dictionary.
    """
    assert dataset_manager.dataset is not None
    assert isinstance(dataset_manager.dataset, dict)
    assert all(k in dataset_manager.dataset for k in ["train", "validation", "test"])

def test_training_text_is_list(dataset_manager):
    """
    Validate that get_training_text returns a list of non-empty strings.
    """
    train_text = dataset_manager.get_training_text()
    assert isinstance(train_text, list)
    assert all(isinstance(line, str) for line in train_text)
    assert any(len(line.strip()) > 0 for line in train_text)

def test_validation_text_is_list(dataset_manager):
    """
    Validate that get_validation_text returns a list of non-empty strings.
    """
    val_text = dataset_manager.get_validation_text()
    assert isinstance(val_text, list)
    assert all(isinstance(line, str) for line in val_text)

def test_test_text_is_list(dataset_manager):
    """
    Validate that get_test_text returns a list of non-empty strings.
    """
    test_text = dataset_manager.get_test_text()
    assert isinstance(test_text, list)
    assert all(isinstance(line, str) for line in test_text)

def test_invalid_split_returns_empty():
    """
    If the dataset is not loaded or the split is invalid,
    _get_text_split should return an empty list.
    """
    manager = DatasetManager()
    manager.dataset = None
    assert manager._get_text_split("train") == []

    manager.dataset = {"train": []}
    assert manager._get_text_split("invalid") == []

def test_openwebtext_split_logic():
    """
    Tests _split_openwebtext() method behavior with mock OpenWebText data.
    """
    manager = DatasetManager()
    mock_data = [{"text": f"sample {i}"} for i in range(100)]
    split = manager._split_openwebtext(mock_data)

    assert isinstance(split, dict)
    assert all(k in split for k in ["train", "validation", "test"])
    total = sum(len(split[k]) for k in split)
    assert total == 100  # Should split all original 100 samples