"""Real-time feature store tests."""

from src.features.online_store import OnlineFeatureStore
from src.streaming.events import Event


def test_set_and_get() -> None:
    store = OnlineFeatureStore()
    store.set_features("C1", {"monetary": 100.0})
    assert store.get("C1")["monetary"] == 100.0
    assert store.get("UNKNOWN") == {}


def test_cart_abandoned_updates_count() -> None:
    store = OnlineFeatureStore()
    event = Event("cart_abandoned", "C1", {"cart_value": 50.0})
    features = store.update_from_event(event)
    assert features["cart_abandoned_count"] == 1
    features = store.update_from_event(event)
    assert features["cart_abandoned_count"] == 2
    assert features["last_cart_value"] == 50.0


def test_order_completed_updates_monetary() -> None:
    store = OnlineFeatureStore()
    event = Event("order_completed", "C1", {"amount": 120.5})
    features = store.update_from_event(event)
    assert features["monetary"] == 120.5
    assert features["order_count"] == 1


def test_session_ended_updates_count() -> None:
    store = OnlineFeatureStore()
    features = store.update_from_event(Event("session_ended", "C1", {}))
    assert features["session_count"] == 1
    assert features["last_event_type"] == "session_ended"
