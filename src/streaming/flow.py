"""Real-time re-engagement flow (streaming consumer).

When an event arrives the feature store is updated; for trigger events (e.g.
cart abandonment) a risk score is computed, a multi-channel message is sent and
the action is tracked in the CDP.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from src.cdp.client import CDPClient
from src.channels.notifier import ChannelNotifier
from src.features.online_store import OnlineFeatureStore
from src.streaming.events import Event, EventConsumer, EventProducer

logger = logging.getLogger(__name__)

TRIGGER_EVENTS = {"cart_abandoned"}


class RealtimeReengagementFlow:
    """Event-driven multi-channel re-engagement pipeline."""

    def __init__(
        self,
        store: OnlineFeatureStore | None = None,
        notifier: ChannelNotifier | None = None,
        cdp: CDPClient | None = None,
        scorer: Callable[[dict], float] | None = None,
    ) -> None:
        self.store = store or OnlineFeatureStore()
        self.notifier = notifier or ChannelNotifier()
        self.cdp = cdp or CDPClient()
        self.scorer = scorer

    def _risk(self, features: dict) -> float:
        if self.scorer is not None:
            return float(self.scorer(features))
        # Rule-based fallback score
        carts = features.get("cart_abandoned_count", 0)
        value = features.get("last_cart_value", 0.0)
        return min(0.95, 0.3 + 0.2 * carts + 0.1 * (value > 100))

    def handle_event(self, event: Event) -> dict:
        features = self.store.update_from_event(event)
        if event.event_type not in TRIGGER_EVENTS:
            return {"customer_id": event.customer_id, "triggered": False}

        risk = self._risk(features)
        cart_value = float(event.properties.get("cart_value", 0.0))
        message = (
            f"Hello, you still have {cart_value:.0f} TRY worth of items in your cart. "
            "Complete your purchase with a special 10% discount!"
        )
        records = self.notifier.send(event.customer_id, message)
        self.cdp.track(
            event.customer_id,
            "reengagement_sent",
            {"risk": risk, "channels": [r.channel for r in records]},
        )
        return {
            "customer_id": event.customer_id,
            "triggered": True,
            "risk": risk,
            "channels": [r.channel for r in records],
            "message": message,
        }

    def run(self, consumer: EventConsumer, max_events: int = 100) -> list[dict]:
        results: list[dict] = []
        for event in consumer.poll(max_events):
            results.append(self.handle_event(event))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming re-engagement demo")
    parser.add_argument("--out", type=Path, default=Path("artifacts/streaming_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    producer = EventProducer()
    demo_events = [
        Event("cart_abandoned", "C1", {"cart_value": 250.0}),
        Event("session_ended", "C2", {}),
        Event("cart_abandoned", "C1", {"cart_value": 80.0}),
        Event("order_completed", "C3", {"amount": 120.0}),
    ]
    for event in demo_events:
        producer.produce(event)

    flow = RealtimeReengagementFlow()
    results = flow.run(EventConsumer(producer))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Report saved: %s", args.out)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
