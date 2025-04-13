"""
Tests for the FileOutputManager.

Verifies:
- update() writes to file
- log_message() appends formatted content
"""

import os
import tempfile
from log_output_manager.file_output_manager import FileOutputManager
from log_output_manager.event_enums import EventType, MetricKey

def test_file_output_training_loss_writes_to_file():
    """
    Test that FileOutputManager writes the formatted log to the file.
    """
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as temp:
        file_path = temp.name
    
    try:
        manager = FileOutputManager(log_file=file_path)
        manager.update(EventType.TRAINING_LOSS, {
            MetricKey.EPOCH: 2,
            MetricKey.STEP: 10,
            MetricKey.LOSS: 1.234
        })

        with open(file_path, 'r') as f:
            lines = f.readlines()
            assert any("[TRAIN] Epoch 2 | Step 10 | Loss: 1.2340" in line for line in lines)
    finally:
        os.remove(file_path)