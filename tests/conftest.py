"""Shared pytest configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep tests independent of external LLM services: pin the provider to mock and
# disable real API keys.
os.environ["LLM_PROVIDER"] = "mock"
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
