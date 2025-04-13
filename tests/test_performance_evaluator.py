"""
Unit tests for the PerformanceEvaluator class.

These tests ensure:
- Observer registration and removal works correctly
- Observers are notified with appropriate event data
- Logging functions trigger expected events
"""

import pytest
from unittest.mock import MagicMock
from performance_evaluator.performance_evaluator import PerformanceEvaluator
from log_output_manager.event_enums import EventType, MetricKey

@pytest.fixture
def evaluator():
    """Fixture to initialize and reset a PerformanceEvaluator instance for each test."""
    evaluator = PerformanceEvaluator()
    evaluator._init_singleton()  # Reset observers
    return evaluator

@pytest.fixture
def mock_observer():
    """Fixture to provide a reusable mock observer for testing."""
    return MagicMock()

def test_add_and_remove_observer(evaluator, mock_observer):
    """Test that observers can be added and removed from the evaluator."""
    evaluator.add_observer(mock_observer)
    assert mock_observer in evaluator.observers

    evaluator.remove_observer(mock_observer)
    assert mock_observer not in evaluator.observers

def test_notify_observers_triggers_update(evaluator, mock_observer):
    """Test that all observers are notified correctly via their update method."""
    evaluator.add_observer(mock_observer)
    test_data = {MetricKey.EPOCH: 1, MetricKey.STEP: 2, MetricKey.LOSS: 0.98}

    evaluator.notify_observers(EventType.TRAINING_LOSS, test_data)

    mock_observer.update.assert_called_once_with(EventType.TRAINING_LOSS, test_data)

def test_log_training_loss_dispatches_event(evaluator, mock_observer):
    """Test that training loss is logged with the correct event and data."""
    evaluator.add_observer(mock_observer)
    evaluator.log_training_loss(epoch=5, step=20, loss=0.321)

    mock_observer.update.assert_called_once_with(
        EventType.TRAINING_LOSS,
        {MetricKey.EPOCH: 5, MetricKey.STEP: 20, MetricKey.LOSS: 0.321}
    )

def test_log_validation_loss_dispatches_event(evaluator, mock_observer):
    """Test that validation loss is logged correctly."""
    evaluator.add_observer(mock_observer)
    evaluator.log_validation_loss(epoch=2, validation_loss=0.12)

    mock_observer.update.assert_called_once_with(
        EventType.VALIDATION_LOSS,
        {MetricKey.EPOCH: 2, MetricKey.LOSS: 0.12}
    )

def test_log_final_summary_dispatches_event(evaluator, mock_observer):
    """Test that the final training summary is sent to observers."""
    evaluator.add_observer(mock_observer)
    evaluator.log_training_final_summary(total_training_time=123.45)

    mock_observer.update.assert_called_once_with(
        EventType.FINAL_SUMMARY,
        {MetricKey.TOTAL_TIME: 123.45}
    )