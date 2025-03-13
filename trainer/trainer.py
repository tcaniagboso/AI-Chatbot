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
import re
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup

from transformer.model import TransformerModel
from transformer.config import batch_size, learning_rate, device, block_size
from dataset_manager.dataset_manager import SingletonDatasetManager
from tokenizer.tokenizer import SingletonTokenizer
from log_output_manager.console_output_manager import ConsoleOutputManager
from log_output_manager.training_metrics_logger import TrainingMetricsLogger
from performance_evaluator.performance_evaluator import SingletonPerformanceEvaluator

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
    def __init__(self, model: TransformerModel, tokenizer: SingletonTokenizer, dataset_manager: SingletonDatasetManager, evaluator: SingletonPerformanceEvaluator):
        """
        Initializes the Trainer.

        Args:
            model (TransformerModel): The Transformer model.
            tokenizer (SingletonTokenizer): Tokenizer for encoding.
            dataset_manager (DatasetManager): Manages dataset loading.
            evaluator (PerformanceEvaluator): Handles logging & evaluation.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset_manager = dataset_manager
        self.evaluator = evaluator
        self.metrics_logger = TrainingMetricsLogger()

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')

        self.dataset_manager.load_pretraining_dataset()

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('checkpoints/latest', exist_ok=True)
        os.makedirs('checkpoints/best', exist_ok=True)
        os.makedirs('logs', exist_ok=True)


    def train(self, num_epochs=3):
        """
        Trains the Transformer model for the specified number of epochs.
        """
        train_loader = self.create_dataloader(self.dataset_manager.get_training_text())
        val_loader = self.create_dataloader(self.dataset_manager.get_validation_text(), shuffle=False)

        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        self.load_latest_checkpoint()

        start_time = time.time()
        for epoch in range(self.current_epoch, num_epochs):
            self.train_one_epoch(epoch, train_loader)

            # Validate & log
            val_loss = self.evaluate_validation_loss(val_loader)
            self.evaluator.log_validation_loss(epoch, val_loss)
            self.metrics_logger.log_validation_loss(epoch, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model()

            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (epoch + 1)) * (num_epochs - (epoch + 1))
            self.evaluator.logger.log_message(f"Estimated time remaining: {remaining_time:.2f} seconds")

        total_training_time = time.time() - start_time
        self.evaluator.logger.log_message(f"Training completed in {total_training_time:.2f} seconds")
        self.metrics_logger.log_final_summary(total_training_time)

    def train_one_epoch(self, epoch, dataloader):
        """
        Trains the model for one epoch.
        """
        self.model.train()

        for step, (xb, yb) in enumerate(dataloader):
            if epoch == self.current_epoch and step < self.current_step:
                continue  # Skip already completed steps if resuming.

            xb, yb = xb.to(device), yb.to(device)

            self.optimizer.zero_grad()
            logits, loss = self.model(xb, yb)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % 10 == 0:
                self.evaluator.log_training_loss(epoch, step, loss.item())
                self.metrics_logger.log_training_loss(epoch, step, loss.item())

            if step % 100 == 0:
                self.save_latest_checkpoint(epoch, step)

        self.current_step = 0
    def evaluate_validation_loss(self, dataloader):
        """
        Computes validation loss.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                logits, loss = self.model(xb, yb)

                total_loss += loss.item() * yb.numel()
                total_tokens += yb.numel()

        return total_loss / total_tokens
    
    def create_dataloader(self, text_samples, shuffle=True):
        """
        Creates a DataLoader for next-token prediction.

        Args:
            text_samples (List[str]): List of text samples.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: Efficient PyTorch DataLoader.
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

    

    def save_latest_checkpoint(self, epoch, step, checkpoint_path = "checkpoints/latest"):
        """
        Saves only the latest checkpoint, overwriting old ones to save storage.
        """

        # Remove old checkpoints
        for f in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, f))

        checkpoint_filename = f"model_epoch{epoch}_step{step}.pt"
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        path = os.path.join(checkpoint_path, checkpoint_filename)
        torch.save(checkpoint, path)
        print(f"Saved latest checkpoint: {path}")
   
    def load_latest_checkpoint(self, checkpoint_path = "checkpoints/latest"):
        """
        Loads the latest checkpoint if available.
        """
        def parse_checkpoint_filename(filename):
            match = re.match(r'model_epoch(\d+)_step(\d+)\.pt', filename)
            return int(match.group(1)), int(match.group(2))
        

        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
        if not checkpoints:
            print("No latest checkpoint found, starting fresh.")
            return

        latest_checkpoint = sorted(checkpoints)[-1]
        epoch, step = parse_checkpoint_filename(latest_checkpoint)
        self.load_checkpoint(os.path.join(checkpoint_path, latest_checkpoint), epoch, step)

    def load_checkpoint(self, path, epoch, step):
        """
        Loads a saved checkpoint.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = epoch
        self.current_step = step + 1
        print(f"Resumed from {path} (epoch {epoch}, step {step})")

    def save_best_model(self, best_path = 'checkpoints/best/best_model.pt'):
        """
        Save current best model (based on validation loss).
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, best_path)
        print("Saved best model (lowest validation loss).")

    def load_best_model(self, best_path = 'checkpoints/best/best_model.pt'):
        """
        Loads the best model for evaluation.
        """
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {best_path}")
        else:
            print("No best model found. Train first.")


if __name__ == '__main__':
    tokenizer = SingletonTokenizer() # Tokenizer
    dataset_manager = SingletonDatasetManager() # DatasetManager

    model = TransformerModel() # Model

    evaluator = SingletonPerformanceEvaluator() # Perormance Evaluator
    evaluator.set_logger(ConsoleOutputManager()) # Set the logger

    print (f"Device: {device}") # Print device

    # Train Model
    trainer = Trainer(model, tokenizer, dataset_manager, evaluator)
    trainer.train(num_epochs=3)

    # Load best Model
    trainer.load_best_model()

    test_loader = trainer.create_dataloader(dataset_manager.get_test_text(), False) # Create test dataloader

    perplexity = evaluator.evaluate_perplexity(model, test_loader, False) # Evaluate Perplexity