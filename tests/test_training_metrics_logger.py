"""
Unit tests for the TrainingMetricsLogger class.

Verifies:
- Training loss, validation loss, final summary, and perplexity are correctly written to CSV.
- update() dispatches based on EventType.
"""

import csv
import os
from log_output_manager.training_metrics_logger import TrainingMetricsLogger
from log_output_manager.event_enums import EventType, MetricKey

def read_csv(filepath):
    """Helper function to read a CSV file as a list of rows."""
    with open(filepath, newline='') as f:
        return list(csv.reader(f))

def test_log_training_loss_creates_and_writes(tmp_path):
    """
    Test that log_training_loss writes a training loss entry to CSV.
    """
    path = tmp_path / "training.csv"
    logger = TrainingMetricsLogger(
        training_filepath=str(path),
        validation_filepath=str(tmp_path / "val.csv"),
        perplexity_filepath=str(tmp_path / "ppl.csv")
    )

    logger.start_timer()
    logger.log_training_loss(epoch=1, step=5, loss=0.789)

    rows = read_csv(path)
    assert rows[0] == ['epoch', 'step_or_phase', 'loss', 'time_elapsed']
    assert rows[1][0] == '1'
    assert rows[1][1] == '5'
    assert rows[1][2] == '0.789'

def test_log_validation_loss_appends(tmp_path):
    """
    Test that log_validation_loss appends a row to validation CSV.
    """
    path = tmp_path / "val.csv"
    logger = TrainingMetricsLogger(
        training_filepath=str(tmp_path / "train.csv"),
        validation_filepath=str(path),
        perplexity_filepath=str(tmp_path / "ppl.csv")
    )

    logger.log_validation_loss(epoch=3, val_loss=1.456)

    rows = read_csv(path)
    assert rows[0] == ['epoch', 'Validation Loss']
    assert rows[1] == ['3', '1.456']

def test_log_perplexity_writes_correctly(tmp_path):
    """
    Test that log_perplexity correctly writes to the perplexity CSV.
    """
    path = tmp_path / "ppl.csv"
    logger = TrainingMetricsLogger(
        training_filepath=str(tmp_path / "train.csv"),
        validation_filepath=str(tmp_path / "val.csv"),
        perplexity_filepath=str(path)
    )

    logger.log_perplexity(model_type="Custom Model", perplexity=12.34)

    rows = read_csv(path)
    assert rows[0] == ['Model Type', 'Perplexity']
    assert rows[1] == ['Custom Model', '12.34']

def test_log_final_summary_appends_training_time(tmp_path):
    """
    Test that log_final_summary writes training completion and time.
    """
    path = tmp_path / "train.csv"
    logger = TrainingMetricsLogger(
        training_filepath=str(path),
        validation_filepath=str(tmp_path / "val.csv"),
        perplexity_filepath=str(tmp_path / "ppl.csv")
    )

    logger.log_final_summary(total_training_time=123.45)

    rows = read_csv(path)
    assert ['Training Completed', '', '', ''] in rows
    assert ['Total Training Time (seconds)', '123.45', '', ''] in rows

def test_update_dispatches_all_event_types(tmp_path):
    """
    Test that update() routes events to the appropriate internal log methods.
    """
    logger = TrainingMetricsLogger(
        training_filepath=str(tmp_path / "train.csv"),
        validation_filepath=str(tmp_path / "val.csv"),
        perplexity_filepath=str(tmp_path / "ppl.csv")
    )

    # Simulate all events
    logger.update(EventType.TRAINING_LOSS, {
        MetricKey.EPOCH: 1,
        MetricKey.STEP: 20,
        MetricKey.LOSS: 0.321
    })

    logger.update(EventType.VALIDATION_LOSS, {
        MetricKey.EPOCH: 1,
        MetricKey.LOSS: 0.789
    })

    logger.update(EventType.FINAL_SUMMARY, {
        MetricKey.TOTAL_TIME: 222.2
    })

    logger.update(EventType.FINAL_PERPLEXITY, {
        MetricKey.MODEL: "Custom",
        MetricKey.PPL: 8.88
    })

    assert os.path.exists(tmp_path / "train.csv")
    assert os.path.exists(tmp_path / "val.csv")
    assert os.path.exists(tmp_path / "ppl.csv")