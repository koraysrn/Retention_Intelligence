"""CDP integration tests."""

from src.cdp.client import CDPClient


def test_track_returns_ok_and_logs() -> None:
    cdp = CDPClient(provider="segment")
    payload = cdp.track("C1", "reengagement_sent", {"risk": 0.5})
    assert payload["status"] == "ok"
    assert payload["provider"] == "segment"
    assert payload["event"] == "reengagement_sent"
    assert len(cdp.events) == 1


def test_identify_returns_traits() -> None:
    cdp = CDPClient(provider="mparticle")
    payload = cdp.identify("C1", {"ltv": 500})
    assert payload["traits"] == {"ltv": 500}
    assert payload["provider"] == "mparticle"
