"""
Trainer module for training the TransformerModel.

This integrates:
- Standard training approach (batch fetching, forward, backward)
- Checkpointing & Best Model Saving
- Cosine Warmup Scheduler
- Logging & Timing
- Training by Epochs
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, List

from transformer.model import TransformerModel
from transformer.config import batch_size, learning_rate, device, block_size
from dataset_manager.dataset_manager import DatasetManager
from tokenizer.tokenizer import Tokenizer
from log_output_manager.console_output_manager import ConsoleOutputManager
from log_output_manager.training_metrics_logger import TrainingMetricsLogger
from performance_evaluator.performance_evaluator import PerformanceEvaluator
from checkpoint_manager.checkpoint_manager import CheckpointManager

# Set a fixed seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# If using GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable for exact reproducibility

print("Random seed set for reproducibility.")

class Trainer:
    """
    Handles training of the Transformer model.

    Responsibilities:
    - DataLoader creation
    - Model training and validation
    - Logging metrics
    - Checkpoint saving/loading via CheckpointManager
    - Scheduler initialization

    Attributes:
        model (TransformerModel): The Transformer model to be trained.
        tokenizer (Tokenizer): Tokenizer used for encoding text samples.
        dataset_manager (DatasetManager): Provides access to training, validation, and test datasets.
        evaluator (PerformanceEvaluator): Logs metrics using the Observer pattern.
        optimizer (AdamW): Optimizer used for training.
        scheduler: Cosine learning rate scheduler with warmup.
        checkpoint_manager (CheckpointManager): Manages saving and loading checkpoints.
        current_epoch (int): Epoch to resume from.
        current_step (int): Step to resume from within the epoch.
        best_val_loss (float): Lowest recorded validation loss.
    """

    def __init__(self, 
                 model: TransformerModel, 
                 tokenizer: Tokenizer, 
                 dataset_manager: DatasetManager, 
                 evaluator: PerformanceEvaluator):
        """
        Initializes the Trainer with model, tokenizer, dataset manager, and evaluator.

        Args:
            model (TransformerModel): Model to train.
            tokenizer (Tokenizer): Tokenizer instance.
            dataset_manager (DatasetManager): Manages datasets.
            evaluator (PerformanceEvaluator): Observer for logging.
        """
        self.model: TransformerModel = model.to(device)
        self.tokenizer: Tokenizer = tokenizer
        self.dataset_manager: DatasetManager = dataset_manager
        self.evaluator: PerformanceEvaluator = evaluator
        self.optimizer: Optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler: Optional[_LRScheduler] = None
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.best_val_loss: float = float('inf')
        self.checkpoint_manager: CheckpointManager = CheckpointManager(self.model, self.optimizer, None)

        self.dataset_manager.load_pretraining_dataset()
    
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('checkpoints/latest', exist_ok=True)
        os.makedirs('checkpoints/best', exist_ok=True)
        os.makedirs('logs', exist_ok=True)


    def train(self, num_epochs: int = 3) -> None:
        """
        Trains the model over a specified number of epochs.

        Args:
            num_epochs (int): Total number of training epochs.
        """

        train_loader = self.create_dataloader(self.dataset_manager.get_training_text())
        val_loader = self.create_dataloader(self.dataset_manager.get_validation_text(), shuffle=False)

        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        self.checkpoint_manager.set_scheduler(self.scheduler)
        self.current_epoch, self.current_step = self.checkpoint_manager.load_latest_checkpoint()

        start_time = time.time()
        for epoch in range(self.current_epoch, num_epochs):
            self.train_one_epoch(epoch, train_loader)

            # Validate & log
            val_loss = self.evaluate_validation_loss(val_loader)
            self.evaluator.log_validation_loss(epoch, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f'New Best Validation Loss: {self.best_val_loss:.4f}')
                self.checkpoint_manager.save_best_model()

        total_training_time = time.time() - start_time
        self.evaluator.log_training_final_summary(total_training_time)

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> None:
        """
        Trains the model for a single epoch.

        Args:
            epoch (int): Current epoch index.
            dataloader (DataLoader): Dataloader providing training data.
        """

        self.model.train()

        for step, (xb, yb) in enumerate(dataloader):
            if epoch == self.current_epoch and step < self.current_step:
                continue  # Skip already completed steps if resuming.

            xb, yb = xb.to(device), yb.to(device)

            self.optimizer.zero_grad()
            _, loss = self.model(xb, yb)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % 10 == 0:
                self.evaluator.log_training_loss(epoch, step, loss.item())

            if step % 100 == 0:
                self.checkpoint_manager.save_latest_checkpoint(epoch, step)

        self.current_step = 0

    def evaluate_validation_loss(self, dataloader: DataLoader) -> float:
        """
        Computes validation loss on a given dataset.

        Args:
            dataloader (DataLoader): Dataloader with validation data.

        Returns:
            float: Average validation loss over the entire dataset.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = self.model(xb, yb)

                total_loss += loss.item() * yb.numel()
                total_tokens += yb.numel()

        return total_loss / total_tokens
    
    def create_dataloader(self, text_samples: List[str], shuffle: bool =True) -> DataLoader:
        """
        Tokenizes input text and prepares a PyTorch DataLoader.

        Args:
            text_samples (List[str]): List of raw text samples.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: Batched dataset of (input, target) token pairs.
        """

        tokenized_data = [self.tokenizer.encode(text)[:block_size] for text in text_samples]

        # Convert to PyTorch tensor and create (input, target) pairs
        input_tensors = []
        target_tensors = []
        
        # Pad sequences to match block_size
        pad_id = self.tokenizer.pad_token_id

        for tokens in tokenized_data:
            if len(tokens) < 2:  # Skip short sequences
                continue
            
            # **Ensure tokens are exactly `block_size`**
            while len(tokens) < block_size + 1:
                tokens.append(pad_id)  # Add padding to make it `block_size+1` long

            input_tensors.append(torch.tensor(tokens[:block_size], dtype=torch.long))  # Full sequence
            target_tensors.append(torch.tensor(tokens[1:block_size+1], dtype=torch.long))  # Shifted sequence
        
        assert all(len(x) == block_size for x in input_tensors), "Mismatch in input size"
        assert all(len(x) == block_size for x in target_tensors), "Mismatch in target size"

        # Convert to tensor and ensure uniform shape
        input_tensors = torch.stack(input_tensors)
        target_tensors = torch.stack(target_tensors)

        dataset = TensorDataset(input_tensors, target_tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    tokenizer = Tokenizer() # Tokenizer
    dataset_manager = DatasetManager() # DatasetManager

    model = TransformerModel() # Model

    evaluator = PerformanceEvaluator() # Perormance Evaluator
    console_output = ConsoleOutputManager()
    csv_output = TrainingMetricsLogger()
    evaluator.add_observer(console_output) # Set the logger
    evaluator.add_observer(csv_output)

    print (f"Device: {device}") # Print device

    # Train Model
    trainer = Trainer(model, tokenizer, dataset_manager, evaluator)
    trainer.train(num_epochs=3)

    # Load best Model
    trainer.checkpoint_manager.load_best_model()

    test_loader = trainer.create_dataloader(dataset_manager.get_test_text(), False) # Create test dataloader
    perplexity = evaluator.evaluate_perplexity(model, test_loader, False) # Evaluate Perplexity