import os
import re
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Tuple
from transformer.model import TransformerModel

class CheckpointManager:
    """
    Facade class that handles saving and loading of model checkpoints.

    Supports:
    - Saving/loading the latest checkpoint (model, optimizer, scheduler)
    - Saving/loading the best model (based on validation loss)
    """

    def __init__(self, model: TransformerModel, optimizer: Optional[Optimizer] = None, scheduler: Optional[_LRScheduler] = None):
        """
        Initializes the CheckpointManager.

        Args:
            model (TransformerModel): The model to manage checkpoints for.
            optimizer (Optional[Optimizer]): Optimizer associated with the model.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        """

        self.model: TransformerModel = model
        self.optimizer: Optional[Optimizer] = optimizer
        self.scheduler: Optional[_LRScheduler] = scheduler

    def set_scheduler(self, scheduler: _LRScheduler) -> None:
        """
        Sets the scheduler after it's created during training.

        Args:
            scheduler (_LRScheduler): Learning rate scheduler.
        """
        self.scheduler = scheduler

    def save_latest_checkpoint(self, epoch: int, step: int, checkpoint_dir: str ="checkpoints/latest") -> None:
        """
        Saves the latest model checkpoint, including optimizer and scheduler states.

        Args:
            epoch (int): Current training epoch.
            step (int): Current training step.
            checkpoint_dir (str): Directory to store the checkpoint.
        """

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Remove existing files
        for f in os.listdir(checkpoint_dir):
            os.remove(os.path.join(checkpoint_dir, f))

        filename = f"model_epoch{epoch}_step{step}.pt"
        path = os.path.join(checkpoint_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None
        }

        torch.save(checkpoint, path)
        print(f"Saved latest checkpoint: {path}")

    def load_latest_checkpoint(self, checkpoint_dir: str ="checkpoints/latest") -> Tuple[Optional[int], Optional[int]]:
        """
        Loads the latest checkpoint from the directory.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.

        Returns:
            Tuple[Optional[int], Optional[int]]: Loaded epoch and step, or (None, None) if not found.
        """

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if not checkpoints:
            print("No latest checkpoint found.")
            return None, None  # No checkpoint

        latest = sorted(checkpoints)[-1]
        path = os.path.join(checkpoint_dir, latest)
        match = re.match(r"model_epoch(\d+)_step(\d+)\.pt", latest)
        epoch, step = int(match.group(1)), int(match.group(2))

        return self.load_checkpoint(path, epoch, step)

    def load_checkpoint(self, path: str, epoch: int, step: int) -> Tuple[int, int]:
        """
        Loads a specific checkpoint into model, optimizer, and scheduler.

        Args:
            path (str): Full path to the checkpoint file.
            epoch (int): Epoch value from filename.
            step (int): Step value from filename.

        Returns:
            Tuple[int, int]: Loaded epoch and next step to resume from.
        """

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Resumed from {path} (epoch {epoch}, step {step})")
        return epoch, step + 1

    def save_best_model(self, best_path: str ="checkpoints/best/best_model.pt") -> None:
        """
        Saves only the model weights as the best model.

        Args:
            best_path (str): Path to store the best model.
        """

        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, best_path)
        print("Saved best model.")

    def load_best_model(self, best_path: str ="checkpoints/best/best_model.pt") -> None:
        """
        Loads the best model from disk.

        Args:
            best_path (str): Path to the saved best model.
        """

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from {best_path}")
        else:
            print("No best model found. You need to train first")