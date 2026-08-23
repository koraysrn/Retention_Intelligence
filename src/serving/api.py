"""Online serving API (FastAPI) + lightweight dashboard.

Provides real-time risk scores for real-time triggers and a simple management
panel. A gateway + authentication is added in enterprise deployments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.llm import LLMClient, mock_chat_reply
from src.config import settings
from src.data.loader import load_ecommerce_data
from src.serving.batch_score import DEFAULT_MODEL_PATH, assign_risk_tier, load_model

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app = FastAPI(title="Churn Risk API", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIST), name="static")

_MODEL = None
_PROFILES = None
_SCORES = None
_MODEL_OUTPUTS = None


def _profiles() -> pd.DataFrame:
    """Return cached raw customer profiles (ecommerce_data.csv)."""
    global _PROFILES
    if _PROFILES is None:
        _PROFILES = load_ecommerce_data(settings.ecommerce_data)
    return _PROFILES


def _scores() -> pd.DataFrame:
    """Return cached batch scores (scores.parquet)."""
    global _SCORES
    if _SCORES is None:
        path = settings.data_processed / "scores.parquet"
        _SCORES = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _SCORES


def _model_outputs() -> pd.DataFrame:
    """Return cached segmentation / CLV / discount model outputs."""
    global _MODEL_OUTPUTS
    if _MODEL_OUTPUTS is None:
        path = settings.data_processed / "customer_models.parquet"
        _MODEL_OUTPUTS = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _MODEL_OUTPUTS


def _fmt_date(value) -> str | None:
    """Convert a datetime/date value to an ISO-8601 string."""
    if value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value)
    return ts.strftime("%Y-%m-%d")


def _risk_explanation(row: dict) -> str:
    """Explains the risk to the end user in one sentence."""
    tier = row.get("risk_tier", "low")
    proba = float(row.get("churn_probability", 0.0) or 0.0)
    if tier == "high":
        return f"This customer has a high churn probability ({proba * 100:.0f}%). Immediate re-engagement is recommended."
    if tier == "medium":
        return f"This customer has a medium churn probability ({proba * 100:.0f}%). Monitor and consider an offer."
    return f"This customer has a low churn probability ({proba * 100:.0f}%). Standard retention is sufficient."


def get_model():
    """Load the model once and cache it."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(DEFAULT_MODEL_PATH)
    return _MODEL


class CustomerRequest(BaseModel):
    customer_id: str
    features: dict[str, Any]


class RiskResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str


class ChatRequest(BaseModel):
    message: str
    customer_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIST / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=RiskResponse)
def predict(req: CustomerRequest) -> RiskResponse:
    """Compute the instantaneous churn probability."""
    model = get_model()
    x = pd.DataFrame([req.features])
    proba = float(model.predict_proba(x)[:, 1][0])
    tier = str(assign_risk_tier(pd.Series([proba])).iloc[0])
    return RiskResponse(customer_id=req.customer_id, churn_probability=proba, risk_tier=tier)


@app.get("/api/summary")
def summary() -> dict:
    """Return dashboard KPI and risk distribution summary."""
    path = settings.data_processed / "scores.parquet"
    if not path.exists():
        return {"available": False}
    df = pd.read_parquet(path)
    dist = df["risk_tier"].value_counts().to_dict()
    return {
        "available": True,
        "total_customers": int(len(df)),
        "avg_churn_probability": round(float(df["churn_probability"].mean()), 4),
        "high_risk_count": int((df["risk_tier"] == "high").sum()),
        "risk_distribution": {str(k): int(v) for k, v in dist.items()},
    }


@app.get("/api/model-metrics")
def model_metrics() -> dict:
    path = settings.artifacts_dir / "model_ecommerce_ensemble" / "metrics.json"
    if not path.exists():
        return {"available": False}
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(df: pd.DataFrame, column: str, top: int = 6) -> list[dict]:
    """Build a segment distribution with share, count and mean churn risk."""
    series = df[column].fillna("Not set").astype(str)
    total = len(series)
    counts = series.value_counts()
    risk = df["_risk"] if "_risk" in df.columns else None

    items: list[dict] = []
    for name, count in counts.head(top).items():
        mask = series == name
        mean_risk = float(risk[mask].mean()) if risk is not None else float("nan")
        items.append(
            {
                "name": name,
                "share": round(float(count / total), 4),
                "count": int(count),
                "risk": round(mean_risk, 3) if not pd.isna(mean_risk) else None,
            }
        )
    return items


def _pct(value: float) -> float:
    return round(float(value) * 100, 1)


@app.get("/api/business-metrics")
def business_metrics() -> dict:
    """Return business metrics and the model-risk linkage over the customer base."""
    if not settings.ecommerce_data.exists():
        return {"available": False}

    df = _profiles().copy()
    df["customer_id"] = df["customer_id"].astype(str)

    scores = _scores()
    if not scores.empty:
        s = scores.copy()
        s["customer_id"] = s["customer_id"].astype(str)
        df = df.merge(s[["customer_id", "churn_probability"]], on="customer_id", how="left")
        df["_risk"] = df["churn_probability"]
    else:
        df["_risk"] = float("nan")

    return {
        "available": True,
        "total_customers": int(len(df)),
        "categories": [
            {
                "id": "sales",
                "name": "Sales & Revenue",
                "metrics": [
                    {"label": "Total Orders", "value": int(df["total_orders"].sum()), "format": "number"},
                    {"label": "Total Spend", "value": round(float(df["total_spend_usd"].sum()), 2), "format": "currency"},
                    {"label": "Avg Basket Value", "value": round(float(df["avg_order_value"].mean()), 2), "format": "currency"},
                    {"label": "Avg Discount Rate", "value": round(float(df["avg_discount_pct"].mean()), 1), "format": "percent"},
                ],
                "distributions": [],
            },
            {
                "id": "profile",
                "name": "Customer Profile",
                "metrics": [
                    {"label": "Repeat Purchase Rate", "value": _pct(df["is_repeat_customer"].mean()), "format": "percent"},
                ],
                "distributions": [
                    {"label": "Demographic Segments", "items": _distribution(df, "age_group")},
                    {"label": "Location Distribution", "items": _distribution(df, "country")},
                    {"label": "Lifetime Value Segments", "items": _distribution(df, "clv_tier")},
                ],
            },
            {
                "id": "engagement",
                "name": "Digital Engagement",
                "metrics": [
                    {"label": "Avg Sessions / Customer", "value": round(float(df["total_sessions"].mean()), 1), "format": "number"},
                    {"label": "Cart Abandonment Rate", "value": _pct(df["has_abandoned_cart"].mean()), "format": "percent"},
                ],
                "distributions": [
                    {"label": "Traffic Device Preference", "items": _distribution(df, "preferred_device_sess")},
                    {"label": "Traffic Source", "items": _distribution(df, "preferred_source_sess")},
                ],
            },
            {
                "id": "experience",
                "name": "Experience & Preferences",
                "metrics": [
                    {"label": "Avg Satisfaction Score", "value": round(float(df["avg_rating_given"].mean()), 2), "format": "number"},
                ],
                "distributions": [
                    {"label": "Payment Habits", "items": _distribution(df, "preferred_payment")},
                    {"label": "Order Device Preference", "items": _distribution(df, "preferred_device_ord")},
                    {"label": "Order Channel", "items": _distribution(df, "preferred_source")},
                    {"label": "Top Categories", "items": _distribution(df, "top_category_bought")},
                ],
            },
        ],
    }


@app.get("/api/experiments")
def experiments() -> dict:
    path = settings.artifacts_dir / "ab_experiment_report.json"
    if not path.exists():
        return {"available": False}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/population")
def population() -> dict:
    """Return population-wide (average) values for the dashboard tiles."""
    if not settings.ecommerce_data.exists():
        return {"available": False}

    df = _profiles().copy()
    df["customer_id"] = df["customer_id"].astype(str)

    outputs = _model_outputs()
    if not outputs.empty:
        m = outputs.copy()
        m["customer_id"] = m["customer_id"].astype(str)
        df = df.merge(m, on="customer_id", how="left")
    else:
        for col in ("segment", "predicted_clv", "predicted_clv_tier", "cart_abandon_probability", "discount_sensitivity"):
            df[col] = None

    def top(column: str) -> str:
        counts = df[column].dropna().astype(str).value_counts()
        return str(counts.index[0]) if len(counts) else ""

    def mean_or_none(column: str, digits: int = 2) -> float | None:
        if df[column].notna().any():
            return round(float(df[column].mean()), digits)
        return None

    return {
        "available": True,
        "total_customers": int(len(df)),
        "age_avg": round(float(df["age"].mean()), 1),
        "total_orders_avg": round(float(df["total_orders"].mean()), 1),
        "total_orders_sum": int(df["total_orders"].sum()),
        "total_spend_sum": round(float(df["total_spend_usd"].sum()), 2),
        "total_spend_avg": round(float(df["total_spend_usd"].mean()), 2),
        "avg_order_value_avg": round(float(df["avg_order_value"].mean()), 2),
        "avg_discount_pct_avg": round(float(df["avg_discount_pct"].mean()), 1),
        "predicted_clv_avg": mean_or_none("predicted_clv"),
        "segment_top": top("segment"),
        "clv_tier_top": top("predicted_clv_tier"),
        "repeat_rate": round(float(df["is_repeat_customer"].mean()) * 100, 1),
        "total_sessions_avg": round(float(df["total_sessions"].mean()), 1),
        "device_sess_top": top("preferred_device_sess"),
        "source_sess_top": top("preferred_source_sess"),
        "cart_abandon_rate": round(float(df["has_abandoned_cart"].mean()) * 100, 1),
        "cart_abandon_prob_avg": (
            round(float(df["cart_abandon_probability"].mean()) * 100, 1)
            if df["cart_abandon_probability"].notna().any()
            else None
        ),
        "payment_top": top("preferred_payment"),
        "device_ord_top": top("preferred_device_ord"),
        "source_ord_top": top("preferred_source"),
        "top_category": top("top_category_bought"),
        "avg_rating_avg": round(float(df["avg_rating_given"].mean()), 2),
        "discount_sensitivity_avg": mean_or_none("discount_sensitivity", 1),
    }


@app.get("/api/insights")
def insights() -> dict:
    """Generate 5 data-driven insights from the customer base + model scores."""
    if not settings.ecommerce_data.exists():
        return {"available": False, "insights": []}

    df = _profiles().copy()
    df["customer_id"] = df["customer_id"].astype(str)

    scores = _scores()
    if not scores.empty:
        s = scores.copy()
        s["customer_id"] = s["customer_id"].astype(str)
        df = df.merge(
            s[["customer_id", "churn_probability", "risk_tier"]],
            on="customer_id",
            how="left",
        )
    else:
        df["churn_probability"] = float("nan")
        df["risk_tier"] = "low"

    items: list[dict] = []
    has_risk = bool(df["churn_probability"].notna().any())

    if has_risk:
        by_age = (
            df.groupby("age_group")["churn_probability"]
            .mean()
            .sort_values(ascending=False)
        )
        if len(by_age):
            items.append(
                {
                    "title": "Highest-risk demographic",
                    "text": (
                        f"The {by_age.index[0]} age group shows the highest average "
                        f"churn risk at {by_age.iloc[0] * 100:.1f}%."
                    ),
                }
            )

    if has_risk:
        by_repeat = df.groupby("is_repeat_customer")["churn_probability"].mean()
        if 0 in by_repeat.index and 1 in by_repeat.index:
            gap = (by_repeat.loc[0] - by_repeat.loc[1]) * 100
            items.append(
                {
                    "title": "Repeat buyers churn less",
                    "text": (
                        f"One-time buyers carry {gap:.1f} percentage points more "
                        "churn risk than repeat buyers."
                    ),
                }
            )

    abandoned_rate = float(df["has_abandoned_cart"].mean()) * 100
    items.append(
        {
            "title": "Cart abandonment",
            "text": (
                f"{abandoned_rate:.1f}% of customers have abandoned a cart — a "
                "direct re-engagement opportunity."
            ),
        }
    )

    top_cat = df["top_category_bought"].value_counts()
    if len(top_cat):
        share = float(top_cat.iloc[0] / len(df)) * 100
        items.append(
            {
                "title": "Most popular category",
                "text": (
                    f"'{top_cat.index[0]}' is the top category, bought by "
                    f"{share:.1f}% of customers."
                ),
            }
        )

    if has_risk:
        high_mask = df["risk_tier"] == "high"
        low_mask = df["risk_tier"] == "low"
        if high_mask.any() and low_mask.any():
            high_disc = float(df.loc[high_mask, "avg_discount_pct"].fillna(0).mean())
            low_disc = float(df.loc[low_mask, "avg_discount_pct"].fillna(0).mean())
            items.append(
                {
                    "title": "Discount dependency",
                    "text": (
                        f"High-risk customers average {high_disc:.1f}% discount vs "
                        f"{low_disc:.1f}% for low-risk customers."
                    ),
                }
            )

    if len(items) < 5:
        avg_rating = float(df["avg_rating_given"].mean())
        items.append(
            {
                "title": "Customer satisfaction",
                "text": f"Average rating given by customers is {avg_rating:.2f} / 5.",
            }
        )

    return {"available": True, "insights": items[:5]}


def _customer_profile(customer_id: str) -> dict:
    """Merge the raw profile + batch score into a rich customer summary."""
    profiles = _profiles()
    mask = profiles["customer_id"].astype(str) == str(customer_id)
    if not mask.any():
        raise HTTPException(status_code=404, detail="Customer not found")

    row = profiles.loc[mask].iloc[0]
    proba = 0.0
    tier = "low"
    scores = _scores()
    if not scores.empty:
        sm = scores[scores["customer_id"].astype(str) == str(customer_id)]
        if not sm.empty:
            proba = float(sm.iloc[0]["churn_probability"])
            tier = str(sm.iloc[0]["risk_tier"])

    segment = ""
    predicted_clv = None
    predicted_clv_tier = ""
    predicted_orders_12m = None
    cart_abandon_probability = None
    discount_sensitivity = None
    outputs = _model_outputs()
    if not outputs.empty:
        om = outputs[outputs["customer_id"].astype(str) == str(customer_id)]
        if not om.empty:
            r = om.iloc[0]
            segment = str(r.get("segment", "") or "")
            predicted_clv_tier = str(r.get("predicted_clv_tier", "") or "")
            if pd.notna(r.get("predicted_clv")):
                predicted_clv = round(float(r["predicted_clv"]), 2)
            if pd.notna(r.get("predicted_orders_12m")):
                predicted_orders_12m = round(float(r["predicted_orders_12m"]), 3)
            if pd.notna(r.get("cart_abandon_probability")):
                cart_abandon_probability = round(float(r["cart_abandon_probability"]), 4)
            if pd.notna(r.get("discount_sensitivity")):
                discount_sensitivity = round(float(r["discount_sensitivity"]), 1)

    profile = {
        "customer_id": str(customer_id),
        "name": str(row.get("name", "")),
        "email": str(row.get("email", "")),
        "country": str(row.get("country", "")),
        "age": int(row.get("age", 0)),
        "age_group": str(row.get("age_group", "")),
        "clv_tier": str(row.get("clv_tier", "")),
        "signup_date": _fmt_date(row.get("signup_date")),
        "first_session_date": _fmt_date(row.get("first_session_date")),
        "total_orders": int(row.get("total_orders", 0)),
        "total_spend_usd": round(float(row.get("total_spend_usd", 0.0)), 2),
        "avg_order_value": round(float(row.get("avg_order_value", 0.0)), 2) if pd.notna(row.get("avg_order_value")) else None,
        "avg_discount_pct": round(float(row.get("avg_discount_pct", 0.0)), 2) if pd.notna(row.get("avg_discount_pct")) else None,
        "total_sessions": int(row.get("total_sessions", 0)),
        "has_abandoned_cart": int(row.get("has_abandoned_cart", 0)),
        "marketing_opt_in": bool(row.get("marketing_opt_in", False)),
        "is_repeat_customer": int(row.get("is_repeat_customer", 0)),
        "top_category_bought": str(row.get("top_category_bought", "")) if pd.notna(row.get("top_category_bought")) else None,
        "preferred_device_ord": str(row.get("preferred_device_ord", "")) if pd.notna(row.get("preferred_device_ord")) else None,
        "preferred_source": str(row.get("preferred_source", "")) if pd.notna(row.get("preferred_source")) else None,
        "preferred_device_sess": str(row.get("preferred_device_sess", "")) if pd.notna(row.get("preferred_device_sess")) else None,
        "preferred_source_sess": str(row.get("preferred_source_sess", "")) if pd.notna(row.get("preferred_source_sess")) else None,
        "preferred_payment": str(row.get("preferred_payment", "")) if pd.notna(row.get("preferred_payment")) else None,
        "avg_rating_given": round(float(row.get("avg_rating_given", 0.0)), 2) if pd.notna(row.get("avg_rating_given")) else None,
        "last_order_date": _fmt_date(row.get("last_order_date")),
        "last_session_date": _fmt_date(row.get("last_session_date")),
        "segment": segment,
        "predicted_clv": predicted_clv,
        "predicted_clv_tier": predicted_clv_tier,
        "predicted_orders_12m": predicted_orders_12m,
        "cart_abandon_probability": cart_abandon_probability,
        "discount_sensitivity": discount_sensitivity,
        "churn_probability": round(proba, 4),
        "risk_tier": tier,
    }
    profile["risk_explanation"] = _risk_explanation(profile)
    return profile


@app.get("/api/predict/{customer_id}")
def predict_customer(customer_id: str) -> dict:
    """Return the real-time risk score for a customer id."""
    profile = _customer_profile(customer_id)
    return {
        "customer_id": profile["customer_id"],
        "churn_probability": profile["churn_probability"],
        "risk_tier": profile["risk_tier"],
        "country": profile["country"],
        "total_spend_usd": profile["total_spend_usd"],
    }


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str) -> dict:
    """Return a rich, end-user-friendly profile for the customer detail panel."""
    return _customer_profile(customer_id)


def _customer_context(customer_id: str | None) -> str:
    """Summarizes the customer profile and score for the chat context."""
    if not customer_id:
        return ""

    try:
        profile = _customer_profile(customer_id)
    except HTTPException:
        return ""

    return (
        f"Customer {profile['customer_id']}; "
        f"country {profile['country']}; age {profile['age']} ({profile['age_group']}); "
        f"CLV tier {profile['clv_tier']}; "
        f"orders {profile['total_orders']}; total spend {profile['total_spend_usd']:,.0f}; "
        f"sessions {profile['total_sessions']}; "
        f"abandoned cart {profile['has_abandoned_cart']}; "
        f"repeat buyer {profile['is_repeat_customer']}; "
        f"churn probability {profile['churn_probability'] * 100:.1f}%; "
        f"risk tier {profile['risk_tier']}"
    )


def _summarize_tool_results(results: list[dict]) -> str:
    lines: list[str] = []
    for result in results:
        if "sent" in result:
            if result.get("sent", 0) == 0:
                lines.append(result.get("message", "No score data found."))
            else:
                lines.append(
                    f"Sent a {result['discount_pct']:g}% discount coupon "
                    f"(valid {result['validity_days']} days) to {result['sent']} high-risk "
                    f"customers via {result['channel']}."
                )
        elif "segments" in result:
            lines.append(
                "Risk segments: "
                + ", ".join(f"{k}: {v}" for k, v in result["segments"].items())
            )
    return "\n".join(lines) or "Action completed."


def _run_agent_loop(message: str, context: str) -> str:
    """Run the multi-step agentic loop that drives the LLM with tools."""
    import json

    from src.agents.actions import TOOL_SCHEMAS, execute_tool

    system = (
        "You are an e-commerce churn analysis and action assistant. "
        "When the user asks for an action (e.g. send a coupon) and parameters are "
        "unclear, proceed with sensible defaults: 15% discount, email channel, "
        "7 days validity. Execute the action without asking. Never share PII."
    )
    user_prompt = f"Question: {message}\n\nCustomer context:\n{context or 'General'}"
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    client = LLMClient()
    results: list[dict] = []

    for _ in range(4):
        resp = client.chat_with_tools(messages, TOOL_SCHEMAS)
        if not resp.tool_calls:
            if resp.content:
                return resp.content
            return _summarize_tool_results(results) if results else mock_chat_reply(message, context)

        assistant_tool_calls = []
        new_results = []
        for idx, call in enumerate(resp.tool_calls):
            call_id = f"call_{idx}"
            result = execute_tool(call["name"], call.get("arguments", {}))
            results.append(result)
            new_results.append(result)
            assistant_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
            )
        messages.append({"role": "assistant", "content": None, "tool_calls": assistant_tool_calls})
        for idx, result in enumerate(new_results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{idx}",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return _summarize_tool_results(results) or "Action completed."


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Question answering or tool-based action (coupon sending, etc.)."""
    context = _customer_context(req.customer_id)
    reply = _run_agent_loop(req.message, context)
    return ChatResponse(reply=reply)


def run() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Churn Risk API")
    parser.add_argument("--host", default=settings.serving_host)
    parser.add_argument("--port", type=int, default=settings.serving_port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
