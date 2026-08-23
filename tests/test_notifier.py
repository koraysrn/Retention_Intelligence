"""Multi-channel communication tests."""

from src.channels.notifier import ChannelNotifier


def test_send_all_enabled_channels() -> None:
    notifier = ChannelNotifier(enabled_channels=["email", "sms", "push"])
    records = notifier.send("C1", "hello")
    assert [r.channel for r in records] == ["email", "sms", "push"]
    assert len(notifier.sent) == 3


def test_send_respects_channels_subset() -> None:
    notifier = ChannelNotifier(enabled_channels=["email", "sms", "push"])
    records = notifier.send("C1", "hello", channels=["push"])
    assert [r.channel for r in records] == ["push"]


def test_disabled_channel_skipped() -> None:
    notifier = ChannelNotifier(enabled_channels=["email"])
    records = notifier.send("C1", "hello", channels=["sms"])
    assert records == []


def test_record_contains_content() -> None:
    notifier = ChannelNotifier(enabled_channels=["email"])
    records = notifier.send("C1", "special offer for you")
    assert records[0].content == "special offer for you"
    assert records[0].customer_id == "C1"
