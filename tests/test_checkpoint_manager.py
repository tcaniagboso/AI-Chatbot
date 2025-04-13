"""
Unit tests for the CheckpointManager class.

These tests verify:
- Saving and loading the latest checkpoint (model + optimizer + scheduler)
- Saving and loading the best model
- File handling and naming conventions
"""

import os
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from checkpoint_manager.checkpoint_manager import CheckpointManager
from transformer.model import TransformerModel

def test_save_and_load_latest_checkpoint(tmp_path):
    """
    Test that a checkpoint can be saved and then loaded correctly.
    """
    model = TransformerModel()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 10)

    manager = CheckpointManager(model, optimizer, scheduler)
    checkpoint_dir = tmp_path / "latest"
    manager.save_latest_checkpoint(epoch=2, step=10, checkpoint_dir=str(checkpoint_dir))

    assert len(list(checkpoint_dir.glob("*.pt"))) == 1

    loaded_epoch, loaded_step = manager.load_latest_checkpoint(str(checkpoint_dir))
    assert loaded_epoch == 2
    assert loaded_step == 11  # step + 1

def test_save_and_load_best_model(tmp_path):
    """
    Test saving and loading the best model weights only (no optimizer/scheduler).
    """
    model = TransformerModel()
    manager = CheckpointManager(model)
    best_model_path = tmp_path / "best" / "best_model.pt"

    # Save
    manager.save_best_model(best_path=str(best_model_path))
    assert os.path.exists(best_model_path)

    # Tamper with model
    for param in model.parameters():
        param.data.zero_()

    # Load
    manager.load_best_model(best_path=str(best_model_path))
    assert not all((param == 0).all() for param in model.parameters()), "Model should have non-zero weights after loading"

def test_load_latest_checkpoint_when_none_exists(tmp_path):
    """
    Test behavior when no checkpoint file exists in the given directory.
    """
    model = TransformerModel()
    manager = CheckpointManager(model)
    checkpoint_dir = tmp_path / "empty_checkpoints"
    checkpoint_dir.mkdir()

    epoch, step = manager.load_latest_checkpoint(str(checkpoint_dir))
    assert epoch is None
    assert step is None

def test_load_best_model_when_missing(tmp_path, capsys):
    """
    Test behavior when best model file is missing.
    """
    model = TransformerModel()
    manager = CheckpointManager(model)
    missing_path = tmp_path / "best" / "non_existent.pt"

    manager.load_best_model(best_path=str(missing_path))
    captured = capsys.readouterr()
    assert "No best model found" in captured.out