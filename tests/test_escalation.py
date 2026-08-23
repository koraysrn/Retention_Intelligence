"""Escalation agent tests."""

from src.agents.escalation import (
    EscalationCase,
    build_summary,
    create_crm_task,
    determine_priority,
    should_escalate,
)


def test_escalate_when_guardrail_fails() -> None:
    assert should_escalate("C1", ltv=500, confidence=0.9, guardrail_passed=False) is True


def test_escalate_when_support_complaints() -> None:
    assert (
        should_escalate("C1", ltv=500, confidence=0.9, guardrail_passed=True, support_complaints=1)
        is True
    )


def test_escalate_high_ltv_low_confidence() -> None:
    assert should_escalate("C1", ltv=2000, confidence=0.5, guardrail_passed=True) is True


def test_no_escalate_normal_customer() -> None:
    assert should_escalate("C1", ltv=500, confidence=0.9, guardrail_passed=True) is False


def test_determine_priority() -> None:
    assert determine_priority(500, 0.9, 0) == "normal"
    assert determine_priority(2000, 0.4, 0) == "high"
    assert determine_priority(500, 0.9, 1) == "high"


def test_build_summary_contains_customer() -> None:
    case = EscalationCase(
        customer_id="C1", reason="guardrail", summary="summary", recommended_action="call"
    )
    text = build_summary(case)
    assert "C1" in text
    assert "guardrail" in text


def test_create_crm_task() -> None:
    case = EscalationCase(
        customer_id="C1",
        reason="x",
        summary="s",
        recommended_action="a",
        priority="high",
    )
    task = create_crm_task(case)
    assert task["task_id"] == "CRM-C1"
    assert task["status"] == "open"
    assert task["priority"] == "high"
