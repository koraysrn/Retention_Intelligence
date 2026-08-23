"""Real-time re-engagement flow tests."""

from src.streaming.events import Event, EventConsumer, EventProducer
from src.streaming.flow import RealtimeReengagementFlow


def test_cart_abandoned_triggers_reengagement() -> None:
    flow = RealtimeReengagementFlow()
    result = flow.handle_event(Event("cart_abandoned", "C1", {"cart_value": 250.0}))
    assert result["triggered"] is True
    assert result["risk"] > 0.3
    assert result["channels"]  # must not be empty
    assert len(flow.notifier.sent) == 3  # 3 default channels
    assert len(flow.cdp.events) == 1


def test_non_trigger_event_no_action() -> None:
    flow = RealtimeReengagementFlow()
    result = flow.handle_event(Event("session_ended", "C2", {}))
    assert result["triggered"] is False
    assert flow.notifier.sent == []
    assert flow.cdp.events == []


def test_scorer_injection_used() -> None:
    flow = RealtimeReengagementFlow(scorer=lambda features: 0.99)
    result = flow.handle_event(Event("cart_abandoned", "C1", {}))
    assert result["risk"] == 0.99


def test_rule_based_risk_increases_with_cart_value() -> None:
    flow = RealtimeReengagementFlow()
    r1 = flow.handle_event(Event("cart_abandoned", "C1", {"cart_value": 10.0}))
    r2 = flow.handle_event(Event("cart_abandoned", "C1", {"cart_value": 500.0}))
    # second event has both an incremented counter and a higher value -> risk increases
    assert r2["risk"] >= r1["risk"]


def test_run_processes_batch() -> None:
    producer = EventProducer()
    producer.produce(Event("cart_abandoned", "C1", {"cart_value": 10.0}))
    producer.produce(Event("session_ended", "C2", {}))
    flow = RealtimeReengagementFlow()
    results = flow.run(EventConsumer(producer))
    assert len(results) == 2
    assert results[0]["triggered"] is True
    assert results[1]["triggered"] is False
