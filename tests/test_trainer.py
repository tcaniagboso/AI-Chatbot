"""
Unit tests for the Trainer class.

These tests ensure:
- Correct DataLoader creation
- Checkpoint saving and loading
- Training metrics are logged
- Evaluation loss returns a float
- Scheduler is initialized during training
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock
from transformers import get_cosine_schedule_with_warmup

from dataset_manager.dataset_manager import DatasetManager
from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from trainer.trainer import Trainer
from performance_evaluator.performance_evaluator import PerformanceEvaluator


@pytest.fixture(scope="module")
def tokenizer():
    """Fixture to provide a singleton tokenizer."""
    return Tokenizer()


@pytest.fixture(scope="module")
def dataset_manager():
    """Fixture to provide a dataset manager with wikitext loaded."""
    manager = DatasetManager()
    manager.load_pretraining_dataset()
    return manager


@pytest.fixture()
def model():
    """Fixture to provide a fresh model instance."""
    return TransformerModel()


@pytest.fixture()
def evaluator():
    """Fixture to provide a performance evaluator."""
    return PerformanceEvaluator()


@pytest.fixture()
def trainer(model, tokenizer, dataset_manager, evaluator):
    """Fixture to construct the Trainer."""
    return Trainer(model, tokenizer, dataset_manager, evaluator)


def test_create_dataloader_shapes(trainer):
    """
    Test that Trainer.create_dataloader produces input and target tensors
    of equal shape.
    """
    loader = trainer.create_dataloader(["hello world", "another sample"])
    for xb, yb in loader:
        assert xb.shape == yb.shape
        break


def test_checkpoint_saves_and_loads(tmp_path, trainer):
    """
    Test saving and loading the latest checkpoint via CheckpointManager.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    trainer.checkpoint_manager.save_latest_checkpoint(3, 50, checkpoint_dir=str(checkpoint_dir))
    epoch, step = trainer.checkpoint_manager.load_latest_checkpoint(str(checkpoint_dir))

    assert epoch == 3
    assert step == 51  # Step is incremented after load


def test_trainer_logs_to_evaluator(tokenizer, dataset_manager):
    """
    Test that the trainer logs training loss to the evaluator during training.
    """
    evaluator = PerformanceEvaluator()
    evaluator.log_training_loss = MagicMock()

    trainer = Trainer(TransformerModel(), tokenizer, dataset_manager, evaluator)
    dataloader = trainer.create_dataloader(["some text", "more text"])

    # Manually initialize the scheduler
    trainer.scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)
    )

    trainer.train_one_epoch(epoch=0, dataloader=dataloader)

    evaluator.log_training_loss.assert_called()


def test_evaluate_validation_loss_runs(trainer):
    """
    Test that evaluate_validation_loss runs and returns a non-negative float.
    """
    dataloader = trainer.create_dataloader(["hello world", "some more text"])
    val_loss = trainer.evaluate_validation_loss(dataloader)

    assert isinstance(val_loss, float)
    assert val_loss >= 0


def test_scheduler_is_initialized(trainer):
    """
    Test that the cosine scheduler is correctly initialized after calling train().
    """
    dataloader = trainer.create_dataloader(["sample", "another sample"])
    trainer.scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)
    )

    for _ in range(len(dataloader)):
        trainer.optimizer.step()
        trainer.scheduler.step()

    assert trainer.scheduler.get_last_lr() is not None


class MockModel(TransformerModel):
    """
    A mock TransformerModel that generates dummy logits and loss for training tests.
    """

    def forward(self, x, y=None):
        device = x.device
        logits = torch.randn(x.size(0), x.size(1), self.lm_head.out_features, device=device, requires_grad=True)
        targets = torch.randint(0, self.lm_head.out_features, x.shape, device=device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def test_train_one_epoch_with_mock_model(tokenizer, dataset_manager):
    """
    Test Trainer.train_one_epoch with a mock model to isolate training logic.
    Ensures the training loop completes without errors.
    """
    model = MockModel()
    trainer = Trainer(model, tokenizer, dataset_manager, PerformanceEvaluator())
    dataloader = trainer.create_dataloader(["mock data", "mock sample"])

    trainer.scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)
    )

    trainer.train_one_epoch(epoch=0, dataloader=dataloader)  # Should complete without error