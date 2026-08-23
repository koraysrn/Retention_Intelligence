"""Agent tools (function calling).

Translates a natural-language command into a concrete action, e.g. "send a
coupon with 10-15% discount valid for 7 days to every risky customer" -> creates
coupons for high-risk customers and dispatches them over multiple channels.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import pandas as pd

from src.channels.notifier import ChannelNotifier
from src.config import settings

MAX_DISCOUNT_PCT = 30.0

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "send_coupon_to_risk_segment",
            "description": (
                "Creates a discount coupon valid for a given number of days and sends "
                "it to high-risk customers through the selected channel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "discount_pct": {
                        "type": "number",
                        "description": "Discount percentage between 10 and 15",
                    },
                    "validity_days": {
                        "type": "integer",
                        "description": "Coupon validity in days (e.g. 7)",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["email", "sms", "push"],
                        "description": "Delivery channel",
                    },
                },
                "required": ["discount_pct", "validity_days", "channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_risk_segments",
            "description": "Lists the number of customers in each risk segment.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _load_scores() -> pd.DataFrame:
    path = settings.data_processed / "scores.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def list_risk_segments(scores: pd.DataFrame | None = None) -> dict:
    df = scores if scores is not None else _load_scores()
    if df.empty:
        return {"available": False, "segments": {}}
    dist = df["risk_tier"].value_counts().to_dict()
    return {"available": True, "segments": {str(k): int(v) for k, v in dist.items()}}


def send_coupon_to_risk_segment(
    discount_pct: float,
    validity_days: int,
    channel: str = "email",
    notifier: ChannelNotifier | None = None,
    scores: pd.DataFrame | None = None,
) -> dict:
    """Create and send coupons to high-risk customers."""
    if not (0 < discount_pct <= MAX_DISCOUNT_PCT):
        raise ValueError(f"Discount must be between 0-%{MAX_DISCOUNT_PCT:.0f}")
    if validity_days <= 0:
        raise ValueError("validity_days must be positive")

    df = scores if scores is not None else _load_scores()
    if df.empty:
        return {"sent": 0, "message": "No score data found; run batch scoring first."}

    high_risk = df[df["risk_tier"] == "high"]
    notifier = notifier or ChannelNotifier()
    codes: list[str] = []

    for customer_id in high_risk["customer_id"].astype(str):
        code = f"COUPON-{uuid.uuid4().hex[:8].upper()}"
        codes.append(code)
        message = (
            f"Your exclusive {discount_pct:g}% discount coupon: {code} "
            f"(valid for {validity_days} days). Use it on the items you last viewed "
            "or left in your cart."
        )
        notifier.send(customer_id, message, channels=[channel])

    return {
        "sent": int(len(high_risk)),
        "discount_pct": float(discount_pct),
        "validity_days": int(validity_days),
        "channel": channel,
        "coupon_codes": codes[:5],
        "total_sent": len(notifier.sent),
    }


def execute_tool(name: str, arguments: dict, **kwargs: Any) -> dict:
    """Translate a tool name and its arguments into a concrete action."""
    if name == "send_coupon_to_risk_segment":
        return send_coupon_to_risk_segment(
            discount_pct=float(arguments.get("discount_pct", 15)),
            validity_days=int(arguments.get("validity_days", 7)),
            channel=str(arguments.get("channel", "email")),
            notifier=kwargs.get("notifier"),
            scores=kwargs.get("scores"),
        )
    if name == "list_risk_segments":
        return list_risk_segments(scores=kwargs.get("scores"))
    raise ValueError(f"Unknown tool: {name}")


def parse_mock_tool_call(message: str) -> list[dict]:
    """Deterministic command parsing when no LLM is available (test/demo fallback)."""
    q = message.lower()
    calls: list[dict] = []

    if ("coupon" in q or "discount" in q or "offer" in q) and (
        "send" in q or "issue" in q or "apply" in q or "trigger" in q
    ):
        m = re.search(r"%?\s*(\d{1,2})\s*[-–]\s*(\d{1,2})", q)
        if m:
            discount = max(int(m.group(1)), int(m.group(2)))
        else:
            m2 = re.search(r"%?\s*(\d{1,2})", q)
            discount = int(m2.group(1)) if m2 else 15
        calls.append(
            {
                "name": "send_coupon_to_risk_segment",
                "arguments": {"discount_pct": discount, "validity_days": 7, "channel": "email"},
            }
        )
    elif "segment" in q or "how many" in q or "list" in q or "breakdown" in q:
        calls.append({"name": "list_risk_segments", "arguments": {}})

    return calls
