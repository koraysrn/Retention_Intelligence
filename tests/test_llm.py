"""LLM provider layer tests."""

import pytest
from src.agents.llm import LLMClient, LLMResponse, build_email_prompt, mask_pii


def test_mock_provider_returns_template() -> None:
    client = LLMClient(provider="mock")
    resp = client.complete("system", "user")
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "mock"
    assert "15% discount" in resp.content


def test_complete_falls_back_without_credentials() -> None:
    # Even when openai is selected, it falls back to mock without an API key
    client = LLMClient(provider="openai")
    resp = client.complete("system", "user")
    assert resp.provider == "mock"


def test_unknown_provider_raises() -> None:
    client = LLMClient(provider="unknown")
    with pytest.raises(ValueError, match="Unknown provider"):
        client.complete("system", "user")


def test_deepseek_provider_falls_back_without_key() -> None:
    client = LLMClient(provider="deepseek")
    resp = client.complete("system", "user")
    # The API key is empty in the test environment -> falls back to mock
    assert resp.provider == "mock"


def test_complete_with_tools_mock_parses() -> None:
    client = LLMClient(provider="mock")
    resp = client.complete_with_tools("sys", "send a coupon to high-risk customers", tools=[])
    assert resp.provider == "mock"
    assert resp.tool_calls
    assert resp.tool_calls[0]["name"] == "send_coupon_to_risk_segment"


def test_mask_pii_email() -> None:
    assert mask_pii("Contact example@mail.com for more information.") == (
        "Contact [EMAIL] for more information."
    )


def test_mask_pii_card() -> None:
    assert mask_pii("Card number 4111111111111111") == "Card number [CARD]"


def test_mask_pii_phone() -> None:
    assert mask_pii("Call us: 555-123-4567") == "Call us: [PHONE]"


def test_build_email_prompt_contains_context() -> None:
    system, user = build_email_prompt({"country": "UK"}, "Sports products campaign", "recency high")
    assert "Sports products campaign" in user
    assert "recency high" in user
    assert "privacy-compliant" in system
