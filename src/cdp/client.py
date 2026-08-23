"""CDP (Segment / mParticle) integration abstraction.

In production this connects to the real CDP APIs (Segment HTTP API, mParticle
SDK); in the prototype events are appended to an in-memory log.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CDPClient:
    """Customer data platform client."""

    def __init__(self, provider: str = "segment", api_key: str = "") -> None:
        self.provider = provider
        self.api_key = api_key
        self.events: list[dict[str, Any]] = []

    def track(self, user_id: str, event: str, properties: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "user_id": user_id,
            "event": event,
            "properties": properties or {},
            "status": "ok",
        }
        self.events.append(payload)
        return payload

    def identify(self, user_id: str, traits: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "user_id": user_id,
            "traits": traits or {},
            "status": "ok",
        }
        return payload
