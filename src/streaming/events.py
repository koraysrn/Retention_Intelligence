"""Kafka event-stream abstraction.

When no real broker (confluent-kafka) is installed, a deterministic in-memory
event bus is used (prototype/test). In an enterprise deployment
``EventProducer``/``EventConsumer`` are replaced with real Kafka clients.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    event_type: str
    customer_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        return cls(
            event_type=data["event_type"],
            customer_id=data["customer_id"],
            properties=data.get("properties", {}),
            timestamp=float(data.get("timestamp", time.time())),
        )


class InMemoryEventBus:
    """In-memory FIFO event bus (test/prototype)."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def produce(self, event: Event) -> None:
        self._events.append(event)

    def poll(self, max_events: int = 100) -> list[Event]:
        batch = self._events[:max_events]
        del self._events[:max_events]
        return batch

    def pending(self) -> int:
        return len(self._events)


class EventProducer:
    """Event producer; uses an in-memory bus when no broker is provided."""

    def __init__(self, broker: str | None = None, bus: InMemoryEventBus | None = None) -> None:
        self.broker = broker
        self._bus = bus or InMemoryEventBus()

    def produce(self, event: Event) -> None:
        if self.broker:
            raise NotImplementedError("Kafka broker integration (confluent-kafka) required")
        self._bus.produce(event)

    @property
    def bus(self) -> InMemoryEventBus:
        return self._bus


class EventConsumer:
    """Event consumer; may share the producer's bus."""

    def __init__(
        self,
        producer: EventProducer | None = None,
        broker: str | None = None,
        bus: InMemoryEventBus | None = None,
    ) -> None:
        self.broker = broker
        self._bus = bus or (producer.bus if producer else InMemoryEventBus())

    def poll(self, max_events: int = 100) -> list[Event]:
        if self.broker:
            raise NotImplementedError("Kafka consumer integration required")
        return self._bus.poll(max_events)
