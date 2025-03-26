import torch 
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from transformer.config import device
from log_output_manager.observer import Observer
from log_output_manager.event_enums import EventType, MetricKey
from singleton.singleton import Singleton
from typing import List, Dict, Any, Tuple

"""
PerformanceEvaluator handles evaluation metrics (like Perplexity)
for both custom TransformerModel and Hugging Face GPT-2 models.
"""

class PerformanceEvaluator(Singleton):
    """
    Handles evaluation-related tasks such as logging training loss
    and calculating perplexity on a dataset.

    Supports both:
    - Custom TransformerModel (decoder-only Transformer built from scratch)
    - Hugging Face pretrained language models (like GPT-2)
    """

    def _init_singleton(self) -> None:
        """
        Initializes the PerformanceEvaluator with an empty list of observers.
        
        This list will be used to notify logging systems or other components 
        when training metrics are updated.
        """
        self.observers: List[Observer] = []

    def add_observer(self, observer: Observer) -> None:
        """
        Registers an observer to receive logging events.

        Parameters
        ----------
        observer : Observer
            An instance of a class implementing the Observer interface.
        """
        self.observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """
        Unregisters an observer from receiving logging events.

        Parameters
        ----------
        observer : Observer
            The observer instance to remove.
        """
        self.observers.remove(observer)

    def notify_observers(self, event_type: EventType, data: Dict[MetricKey, Any]) -> None:
        """
        Notifies all registered observers of an event.

        Parameters
        ----------
        event_type : EventType
            Type of the event (e.g., "training_loss", "validation_loss").
        data : dict
            A dictionary containing the relevant data for the event.
        """
        for observer in self.observers:
            observer.update(event_type, data)

    def log_training_loss(self, epoch: int, step: int, loss: float) -> None:
        """
        Logs and notifies observers of training loss.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        step : int
            The current step or batch number.
        loss : float
            The loss value at this step.
        """
        self.notify_observers(EventType.TRAINING_LOSS, {MetricKey.EPOCH: epoch, MetricKey.STEP: step, MetricKey.LOSS: loss})

    def log_validation_loss(self, epoch: int, validation_loss: float) -> None:
        """
        Logs and notifies observers of validation loss.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        validation_loss : float
            The loss value computed on the validation set.
        """
        self.notify_observers(EventType.VALIDATION_LOSS, {MetricKey.EPOCH: epoch, MetricKey.LOSS: validation_loss})


    def log_training_final_summary(self, total_training_time: float) -> None:
        """
        Logs and notifies observers of training final summary.

        Parameters
        ----------
        total_training_time : float
            The total training time in seconds.
        """
        self.notify_observers(EventType.FINAL_SUMMARY, {MetricKey.TOTAL_TIME: total_training_time})


    def evaluate_perplexity(self, model, dataloader: DataLoader, is_huggingface_model: bool = False) -> float:
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

        model_type = "Hugging Face Model" if is_huggingface_model else "Custom Model"

        self.notify_observers(EventType.EVALUATING_PERPLEXITY, {MetricKey.MODEL: model_type})

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

                        if isinstance(output, Tuple):  
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

        self.notify_observers(EventType.FINAL_PERPLEXITY, {MetricKey.PPL: perplexity, MetricKey.MODEL: model_type})

        return perplexity