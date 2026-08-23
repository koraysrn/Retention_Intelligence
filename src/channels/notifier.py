"""Multi-channel communication layer (email, sms, push).

In production this connects to ESP/SMS gateway/push providers; in the prototype
sent messages are appended to an in-memory log.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_CHANNELS = ("email", "sms", "push")


@dataclass
class MessageRecord:
    customer_id: str
    channel: str
    content: str

    def to_dict(self) -> dict:
        return self.__dict__


class ChannelNotifier:
    """Send a message to a customer over multiple channels."""

    def __init__(self, enabled_channels: list[str] | None = None) -> None:
        self.enabled_channels = enabled_channels or ["email", "sms", "push"]
        self.sent: list[MessageRecord] = []

    def send(
        self,
        customer_id: str,
        message: str,
        channels: list[str] | None = None,
    ) -> list[MessageRecord]:
        targets = channels or self.enabled_channels
        records: list[MessageRecord] = []
        for channel in targets:
            if channel not in self.enabled_channels:
                continue
            record = MessageRecord(customer_id=customer_id, channel=channel, content=message)
            self.sent.append(record)
            records.append(record)
        return records
