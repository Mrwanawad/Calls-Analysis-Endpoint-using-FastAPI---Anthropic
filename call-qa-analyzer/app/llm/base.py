"""Abstract LLM provider interface for swappable model backends."""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models import CallAnalysisResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMUsageStats:
    """Token usage and latency stats for observability."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


@dataclass
class LLMResult:
    """Wraps parsed analysis + raw metadata for logging."""

    analysis: CallAnalysisResponse
    usage: LLMUsageStats = field(default_factory=LLMUsageStats)
    raw_response: str = ""


class LLMProvider(abc.ABC):
    """Abstract base class that all LLM providers must implement."""

    @abc.abstractmethod
    async def analyze_transcript(
        self, system_prompt: str, user_prompt: str
    ) -> LLMResult:
        """Send prompts to the LLM and return a structured CallAnalysisResponse."""

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            "LLM call failed (attempt %d), retrying: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        ),
    )
    async def _call_with_retry(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Wrapper that subclasses call internally; handles retry logic."""
        start = time.perf_counter()
        result = await self._raw_call(system_prompt, user_prompt)
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        return result

    @abc.abstractmethod
    async def _raw_call(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Provider-specific API call. Returns dict with keys:
        content (str), prompt_tokens (int), completion_tokens (int), model (str).
        """

    @staticmethod
    def _parse_response(raw_content: str) -> CallAnalysisResponse:
        """Parse LLM JSON output into the Pydantic model, stripping markdown fences."""
        text = raw_content.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        return CallAnalysisResponse.model_validate(data)
