"""Orchestrator end-to-end flow tests."""

from src.agents.orchestrator import run_workflow


def test_workflow_sends_for_normal_customer() -> None:
    profile = {
        "monetary": 200.0,
        "churn_probability": 0.9,
        "support_complaints": 0,
        "country": "UK",
    }
    result = run_workflow("C1", profile=profile)
    assert result["decision"] == "SENT"
    assert result["escalated"] is False
    assert result["guardrail_passed"] is True
    assert result["final_email"]
    assert "15%" in result["final_email"]


def test_workflow_escalates_high_ltv_low_confidence() -> None:
    profile = {"monetary": 2000.0, "churn_probability": 0.5, "support_complaints": 0}
    result = run_workflow("C2", profile=profile)
    assert result["decision"] == "ESCALATED"
    assert result["escalated"] is True
    assert result["escalation_summary"]
    assert "C2" in result["escalation_summary"]


def test_workflow_deterministic() -> None:
    profile = {"monetary": 200.0, "churn_probability": 0.9}
    r1 = run_workflow("C3", profile=profile)
    r2 = run_workflow("C3", profile=profile)
    assert r1["final_email"] == r2["final_email"]
    assert r1["decision"] == r2["decision"]
