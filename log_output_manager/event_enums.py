from enum import Enum

class EventType(Enum):
    """
    Enumeration of event types for observer notifications.

    This enum defines all supported event types that can be emitted by
    the PerformanceEvaluator during training and evaluation.

    Attributes:
        TRAINING_LOSS: Indicates training loss was computed at a step.
        VALIDATION_LOSS: Indicates validation loss was computed after an epoch.
        FINAL_SUMMARY: Indicates training has completed with a final summary.
        EVALUATING_PERPLEXITY: Indicates that perplexity evaluation is starting.
        FINAL_PERPLEXITY: Indicates that perplexity has been computed.
    """

    TRAINING_LOSS = "training_loss"
    VALIDATION_LOSS = "validation_loss"
    FINAL_SUMMARY = "final_summary"
    FINAL_PERPLEXITY = "final_perplexity"
    EVALUATING_PERPLEXITY = "evaluating_perplexity"
    UNKNOWN = "UNKNOWN EVENT"

class MetricKey(Enum):
    """
    Enumeration of standard keys used in metric event data.

    These keys represent the fields expected in the data dictionaries
    passed to observers (e.g., epoch number, step number, loss, etc.).

    Attributes:
        EPOCH: Epoch number in training or evaluation.
        STEP: Current batch or step within an epoch.
        LOSS: Loss value at a given point.
        TOTAL_TIME: Total training time in seconds.
        PPL: Perplexity score.
        MODEL: Name or type of the model being evaluated.
    """
    
    EPOCH = "epoch"
    STEP = "step"
    LOSS = "loss"
    TOTAL_TIME = "total_time"
    PPL = "ppl"
    MODEL = "model"