from abc import ABC, abstractmethod
from typing import Dict, Any

from log_output_manager.event_enums import EventType, MetricKey

class Observer(ABC):
    """
    Abstract base class representing an Observer in the Observer design pattern.

    Classes that inherit from Observer must implement the `update` method to
    handle notifications from the Subject (Observable).

    This is typically used to decouple event producers from event consumers,
    allowing multiple observers to react to changes or events in a flexible and scalable way.
    """
     
    @abstractmethod
    def update(self, event_type: EventType, data: Dict[MetricKey, Any]) -> None:
        """
        Handles an update notification from the Subject.

        Args:
            event_type (EventType): An enum representing the type of event that occurred.
                              Example: EventType.TRAINING_LOSS, EventType.VALIDATION_LOSS, EventType.FINAL_SUMMARY.
            data (Dict[MetricKey, Any]): A dictionary containing relevant data for the event.
                         The structure of this dictionary depends on the event_type.

        Example:
            observer.update(EventType.TRAINING_LOSS, {MetricKey.EPOCH: 1, MetricKey.STEP: 100, MetricKey.LOSS: 2.14})
        """
        pass