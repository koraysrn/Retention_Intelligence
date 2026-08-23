"""Streaming event-flow tests."""

from src.streaming.events import Event, EventConsumer, EventProducer, InMemoryEventBus


def test_event_roundtrip() -> None:
    e = Event("cart_abandoned", "C1", {"cart_value": 10.0}, timestamp=1.0)
    d = e.to_dict()
    e2 = Event.from_dict(d)
    assert e2.event_type == "cart_abandoned"
    assert e2.customer_id == "C1"
    assert e2.properties == {"cart_value": 10.0}
    assert e2.timestamp == 1.0


def test_in_memory_producer_consumer_roundtrip() -> None:
    producer = EventProducer()
    producer.produce(Event("a", "C1"))
    producer.produce(Event("b", "C2"))
    consumer = EventConsumer(producer)
    batch = consumer.poll()
    assert [e.event_type for e in batch] == ["a", "b"]


def test_consumer_poll_drains_bus() -> None:
    bus = InMemoryEventBus()
    bus.produce(Event("x", "C1"))
    bus.produce(Event("y", "C1"))
    consumer = EventConsumer(bus=bus)
    assert len(consumer.poll(1)) == 1
    assert bus.pending() == 1


def test_broker_producer_raises_not_implemented() -> None:
    import pytest

    producer = EventProducer(broker="localhost:9092")
    with pytest.raises(NotImplementedError):
        producer.produce(Event("a", "C1"))
