# utils/message_bus.py


from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import time


@dataclass
class Message:
    """A message that gets sent through the bus"""
    topic: str
    data: Any
    sender: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MessageBus:
    """Simple message bus implementation"""

    def __init__(self):
        # Dictionary to store subscribers for each topic
        # Format: {"topic_name": [subscriber_function1, subscriber_function2, ...]}
        self._subscribers: Dict[str, List[Callable]] = {}

        # Optional: store message history for debugging
        self._message_history: List[Message] = []

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe a function to receive messages on a topic"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []

        self._subscribers[topic].append(callback)
        print(f"Subscribed to '{topic}'")

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe a function from a topic"""
        if topic in self._subscribers:
            self._subscribers[topic].remove(callback)

    def publish(self, topic: str, data: Any, sender: str = "unknown"):
        """Send a message to all subscribers of a topic"""
        message = Message(topic=topic, data=data, sender=sender)

        # Store in history (optional)
        self._message_history.append(message)

        # Send to all subscribers
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"Error calling subscriber for {topic}: {e}")

        print(f"Published '{topic}' from {sender}")

    def get_message_history(self):
        """Get all messages sent through the bus"""
        return self._message_history
    def cleanup(self):
        """Clean up message bus resources"""
        try:
            self._subscribers.clear()
            self._message_history.clear()
            print("✓ Message bus cleanup completed")
        except Exception as e:
            print(f"Error cleaning up message bus: {e}")
