"""LLM provider abstraction layer.

Switch between OpenAI / Anthropic / Ollama via an environment variable. When no
API key is present or the call fails, a deterministic mock template is used
(prototype / test / demo). Azure / Bedrock are easy to add in enterprise
deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tool_calls: list = field(default_factory=list)


MOCK_EMAIL_TEMPLATE = (
    "Hello, we prepared a 15% discount just for you based on your past purchases. "
    "Check your cart before it's gone — your exclusive offer is waiting!"
)

SUPPORTED_PROVIDERS = {"openai", "anthropic", "ollama", "deepseek", "mock"}


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Text generation; falls back to the mock template when credentials are missing."""
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider: {self.provider}")
        if self.provider == "mock" or not self._has_credentials():
            return self._mock_complete(user_prompt)

        if self.provider == "openai":
            return self._openai_complete(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._anthropic_complete(system_prompt, user_prompt)
        if self.provider == "ollama":
            return self._ollama_complete(system_prompt, user_prompt)
        if self.provider == "deepseek":
            return self._deepseek_complete(system_prompt, user_prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _has_credentials(self) -> bool:
        if self.provider == "openai":
            return bool(settings.openai_api_key)
        if self.provider == "anthropic":
            return bool(settings.anthropic_api_key)
        if self.provider == "deepseek":
            return bool(settings.deepseek_api_key)
        return self.provider == "ollama"

    def _mock_complete(self, user_prompt: str) -> LLMResponse:
        return LLMResponse(content=MOCK_EMAIL_TEMPLATE, provider="mock", model="mock-template")

    def _openai_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            return LLMResponse(content=content, provider="openai", model=settings.openai_model)
        except Exception as exc:  # noqa: BLE001 — fallback is mandatory
            logger.warning("OpenAI call failed, fell back to mock: %s", exc)
            return self._mock_complete(user_prompt)

    def _anthropic_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
            return LLMResponse(content=content, provider="anthropic", model=settings.anthropic_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic call failed, fell back to mock: %s", exc)
            return self._mock_complete(user_prompt)

    def _deepseek_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            resp = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            return LLMResponse(
                content=content, provider="deepseek", model=settings.deepseek_model
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek call failed, fell back to mock: %s", exc)
            return self._mock_complete(user_prompt)

    def complete_with_tools(self, system_prompt: str, user_prompt: str, tools: list) -> LLMResponse:
        """Single-step tool-enabled (function calling) text generation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_with_tools(messages, tools)

    def chat_with_tools(self, messages: list[dict], tools: list) -> LLMResponse:
        """Tool-enabled call with the full message list (for the agentic loop)."""
        if self.provider == "mock" or not self._has_credentials():
            from src.agents.actions import parse_mock_tool_call

            # If a tool result has already been returned, stop calling tools and end the loop.
            if any(m.get("role") == "tool" for m in messages):
                return LLMResponse(content="", provider="mock", model="mock-tools", tool_calls=[])
            user_text = " ".join(
                m.get("content") or "" for m in messages if m.get("role") == "user"
            )
            return LLMResponse(
                content="",
                provider="mock",
                model="mock-tools",
                tool_calls=parse_mock_tool_call(user_text),
            )
        if self.provider in ("deepseek", "openai"):
            return self._openai_compatible_messages(messages, tools)
        raise ValueError(f"{self.provider} does not support tool calls")

    def _openai_compatible_messages(self, messages: list[dict], tools: list) -> LLMResponse:
        try:
            import json

            from openai import OpenAI

            if self.provider == "deepseek":
                client = OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
                model = settings.deepseek_model
            else:
                client = OpenAI(api_key=settings.openai_api_key)
                model = settings.openai_model

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            message = resp.choices[0].message
            calls: list[dict] = []
            for tc in getattr(message, "tool_calls", None) or []:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:  # noqa: BLE001
                    args = {}
                calls.append({"name": tc.function.name, "arguments": args})
            return LLMResponse(
                content=message.content or "",
                provider=self.provider,
                model=model,
                tool_calls=calls,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool call failed, fell back to deterministic parsing: %s", exc)
            from src.agents.actions import parse_mock_tool_call

            user_text = " ".join(
                m.get("content") or "" for m in messages if m.get("role") == "user"
            )
            return LLMResponse(
                content="",
                provider="mock",
                model="mock-tools",
                tool_calls=parse_mock_tool_call(user_text),
            )

    def _ollama_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            import requests

            resp = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": "llama3.2",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            return LLMResponse(content=content, provider="ollama", model="llama3.2")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama call failed, fell back to mock: %s", exc)
            return self._mock_complete(user_prompt)


def mask_pii(text: str) -> str:
    """PII masking (KVKK/GDPR). Raw PII is never sent to the LLM.

    Prototype: simple regex-based masking. Production: PII Vault integration.
    """
    import re

    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    text = re.sub(r"\b\d{10,16}\b", "[CARD]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    return text


def build_email_prompt(profile: dict, context: str, risk_reason: str) -> tuple[str, str]:
    """Builds a (system, user) prompt pair for a personalized win-back email."""
    system = (
        "You are an e-commerce retention email writer. Be friendly, concise, "
        "personalized and privacy-compliant. Never share PII."
    )
    user = (
        f"Customer profile: {profile}\n"
        f"Churn risk reason: {risk_reason}\n"
        f"Relevant context: {context}\n\n"
        "Write a personalized win-back email for this customer that does not exceed the discount limit."
    )
    return system, user


def build_chat_prompt(question: str, context: str = "") -> tuple[str, str]:
    """Builds a (system, user) prompt pair for the analytics assistant chat."""
    system = (
        "You are an e-commerce churn analytics assistant. Answer concisely, clearly "
        "and action-oriented based on customer data. Never share PII."
    )
    user = f"Question: {question}\n\nCustomer context:\n{context or 'General question'}"
    return system, user


def mock_chat_reply(question: str, context: str = "") -> str:
    """Deterministic, meaningful reply used when no LLM key is available."""
    q = question.lower()
    if "risk" in q and any(k in q for k in ("why", "reason", "driver", "high", "cause", "factor")):
        reply = (
            "Risk is high because the customer has not purchased for a long time and "
            "their purchase cadence has declined. The strongest signals are recency "
            "(time since the last purchase) and purchase frequency."
        )
        return f"{reply} {context}" if context else reply
    if any(k in q for k in ("reduce", "decrease", "action", "what should", "what can", "win back")):
        return (
            "To reduce risk: send personalized discounts or offers, trigger cart "
            "reminders, route high-LTV customers to a sales representative, and "
            "measure campaign impact with an A/B test."
        )
    return (
        "I can help with customer risk, risk reduction, or campaign recommendations. "
        "Try asking about churn drivers or win-back targeting."
    )
