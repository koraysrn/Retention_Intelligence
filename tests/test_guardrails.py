"""Guardrail check tests."""

from src.agents.guardrails import llm_judge, run_guardrails


def test_clean_content_passes() -> None:
    result = run_guardrails(
        "Hello Jane, we applied a 15% discount to the items in your cart. Have a great day!"
    )
    assert result.passed is True


def test_excessive_discount_fails() -> None:
    result = run_guardrails("You've earned a special 60% discount!")
    assert result.passed is False
    assert "discount_within_limit" in result.reasons


def test_pii_email_fails() -> None:
    result = run_guardrails("Contact example@mail.com for more information.")
    assert result.passed is False
    assert "no_pii_leak" in result.reasons


def test_pii_phone_fails() -> None:
    result = run_guardrails("Call us: 555-123-4567")
    assert result.passed is False
    assert "no_pii_leak" in result.reasons


def test_forbidden_term_fails() -> None:
    result = run_guardrails("We are giving you an unlimited free trial")
    assert result.passed is False
    assert "no_forbidden_terms" in result.reasons


def test_llm_judge_grounded() -> None:
    assert llm_judge("A special discount offer for you", "") is True
    assert llm_judge("This text is completely unrelated", "", grounded_terms=["discount"]) is False
