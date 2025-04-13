"""
Tests for the ConsoleOutputManager class.

Verifies that:
- update() handles EventType correctly.
- log_message() prints expected messages.
"""

from log_output_manager.console_output_manager import ConsoleOutputManager
from log_output_manager.event_enums import EventType, MetricKey

def test_console_output_training_loss(capfd):
    """
    Test that ConsoleOutputManager logs formatted training loss to stdout.
    """
    manager = ConsoleOutputManager()
    
    event_data = {
        MetricKey.EPOCH: 1,
        MetricKey.STEP: 50,
        MetricKey.LOSS: 0.245
    }

    manager.update(EventType.TRAINING_LOSS, event_data)

    out, _ = capfd.readouterr()
    assert "[TRAIN] Epoch 1 | Step 50 | Loss: 0.2450" in out