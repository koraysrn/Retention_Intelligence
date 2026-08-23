"""Escalation agent: detecting and summarizing cases that need human intervention.

Opens a task for the sales representative via the CRM and produces a summary
report containing the customer profile, risk reason and recommended action.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EscalationCase:
    customer_id: str
    reason: str
    summary: str
    recommended_action: str
    priority: str = "normal"


def should_escalate(
    customer_id: str,
    ltv: float,
    confidence: float,
    guardrail_passed: bool,
    support_complaints: int = 0,
    high_ltv_threshold: float = 1000.0,
    confidence_threshold: float = 0.8,
) -> bool:
    """Decide whether human intervention is required.

    Rules (docs/agentic_ai_design.md, section 5):
    - High LTV and low confidence
    - Guardrail failure
    - An existing support complaint
    """
    if not guardrail_passed:
        return True
    if support_complaints > 0:
        return True
    return ltv >= high_ltv_threshold and confidence < confidence_threshold


def determine_priority(
    ltv: float,
    confidence: float,
    support_complaints: int,
    high_ltv_threshold: float = 1000.0,
) -> str:
    """Determine the escalation priority."""
    if support_complaints > 0:
        return "high"
    if ltv >= high_ltv_threshold and confidence < 0.5:
        return "high"
    return "normal"


def build_summary(case: EscalationCase) -> str:
    """Build the summary text to be forwarded to the sales representative."""
    return (
        f"Customer: {case.customer_id}\n"
        f"Priority: {case.priority}\n"
        f"Reason: {case.reason}\n"
        f"Recommended action: {case.recommended_action}\n\n"
        f"Summary:\n{case.summary}"
    )


def create_crm_task(case: EscalationCase) -> dict:
    """Represent the task to be opened in the CRM (returns a local dict in the prototype)."""
    return {
        "task_id": f"CRM-{case.customer_id}",
        "status": "open",
        "priority": case.priority,
        "summary": build_summary(case),
    }
