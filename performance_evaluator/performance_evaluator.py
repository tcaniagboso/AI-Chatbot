import torch 
from tqdm import tqdm
import math
from transformer.config import device
from log_output_manager.log_output_manager import LogOutputManager
from singleton.singleton import Singleton

"""
PerformanceEvaluator handles evaluation metrics (like Perplexity)
for both custom TransformerModel and Hugging Face GPT-2 models.
"""

class PerformanceEvaluator:
    """
    Handles evaluation-related tasks such as logging training loss
    and calculating perplexity on a dataset.

    Supports both:
    - Custom TransformerModel (decoder-only Transformer built from scratch)
    - Hugging Face pretrained language models (like GPT-2)
    """

    def __init__(self):
        """Initializes the evaluator with an optional logger."""
        self.logger = None

    def set_logger(self, logger: LogOutputManager):
        """
        Sets the logger for the evaluator.

        Args:
            logger (LogOutputManager): Log manager for logging metrics.
        """
        self.logger = logger

    def log_training_loss(self, epoch: int, step: int, loss: float):
        """
        Logs the training loss at regular intervals.

        Args:
            epoch (int): Current epoch number.
            step (int): Current batch/step number within the epoch.
            loss (float): Current loss value.
        """
        self.logger.log_message(f"[TRAIN] Epoch {epoch} | Step {step} | Loss: {loss:.4f}")

    def log_validation_loss(self, epoch: int, validation_loss: float):
        """
        Logs the validation loss at regular intervals.

        Args:
            epoch (int): Current epoch number.
            validation_loss (float): Current validation loss value.
        """
        self.logger.log_message(f"[VALIDATION] Epoch {epoch} | Validation Loss: {validation_loss:.4f}")


    def evaluate_perplexity(self, model, dataloader, is_huggingface_model: bool = False) -> float:
        """
        Evaluates the model on a given dataset and calculates perplexity.

        Args:
            model: Either custom TransformerModel or Hugging Face GPT-2.
            dataloader: DataLoader providing tokenized text batches.
            is_huggingface_model (bool): If True, assumes Hugging Face model structure.

        Returns:
            float: Perplexity score.
        """
        model.to(device)
        model.eval()

        if self.logger:
            model_type = "Hugging Face GPT-2" if is_huggingface_model else "Custom Transformer Model"
            self.logger.log_message(f"Evaluating Perplexity on {model_type}...")

        total_nll = 0.0
        total_tokens = 0
        stride = 64
        max_length = 128

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):

                # Extract `input_ids` properly
                if is_huggingface_model:
                    input_ids = batch['input_ids'].to(device)
                else:
                    input_ids = torch.stack(batch).to(device)

                seq_len = input_ids.size(1)
                prev_end_loc = 0

                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc
                    chunk = input_ids[:, begin_loc:end_loc]
                    target_ids = chunk.clone()

                    if chunk.dim() == 3:
                        chunk = chunk.view(-1, chunk.size(-1))  # Flatten batch & sequence dim

                    # Mask the overlapping tokens in the target_ids to avoid duplicate loss calculation
                    if trg_len < chunk.size(1):
                        target_ids[:, :-trg_len] = -100  

                    # Pass through model
                    if is_huggingface_model:
                        outputs = model(input_ids=chunk, labels=target_ids)
                        nll = outputs.loss
                    else:
                        output = model(chunk)  # Ensure only `input_ids` is passed

                        if isinstance(output, tuple):  
                            logits = output[0]  # Extract logits if model outputs (logits, loss)
                        else:
                            logits = output  # If only logits, use as is

                        # Compute loss
                        nll = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),  
                            target_ids.view(-1),
                            ignore_index=-100,
                            reduction='mean'
                        )

                    total_nll += nll.cpu().item() * trg_len
                    total_tokens += trg_len

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

        # Avoid division by zero
        if total_tokens == 0:
            return float("inf")  

        avg_nll = total_nll / total_tokens
        perplexity = math.exp(avg_nll)

        if self.logger:
            model_type = "Hugging Face Model" if is_huggingface_model else "Custom Model"
            self.logger.log_message(f"Final Perplexity on Test Set ({model_type}): {perplexity:.4f}")

        return perplexity


class SingletonPerformanceEvaluator(PerformanceEvaluator, Singleton):
    """
    Singleton wrapper for PerformanceEvaluator.
    """

    def _init_singleton(self):
        """Ensures proper initialization."""
        super().__init__()