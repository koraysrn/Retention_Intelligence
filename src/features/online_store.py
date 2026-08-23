"""Real-time feature store (online feature repository).

Production uses Redis/Feast; the prototype uses an in-memory dict. Customer
features are updated on the fly as events arrive.
"""

from __future__ import annotations

from typing import Any

from src.streaming.events import Event


class OnlineFeatureStore:
    """Online feature store fed by incoming events."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, customer_id: str) -> dict[str, Any]:
        return dict(self._store.get(customer_id, {}))

    def set_features(self, customer_id: str, features: dict[str, Any]) -> None:
        self._store.setdefault(customer_id, {}).update(features)

    def update_from_event(self, event: Event) -> dict[str, Any]:
        """Apply an event to customer features and return the updated state."""
        cid = event.customer_id
        feats = self.get(cid)
        feats["last_event_type"] = event.event_type
        feats["last_event_at"] = event.timestamp

        if event.event_type == "cart_abandoned":
            feats["cart_abandoned_count"] = feats.get("cart_abandoned_count", 0) + 1
            feats["last_cart_value"] = float(event.properties.get("cart_value", 0.0))
        elif event.event_type == "order_completed":
            feats["order_count"] = feats.get("order_count", 0) + 1
            feats["monetary"] = feats.get("monetary", 0.0) + float(
                event.properties.get("amount", 0.0)
            )
        elif event.event_type == "session_ended":
            feats["session_count"] = feats.get("session_count", 0) + 1

        self.set_features(cid, feats)
        return feats
