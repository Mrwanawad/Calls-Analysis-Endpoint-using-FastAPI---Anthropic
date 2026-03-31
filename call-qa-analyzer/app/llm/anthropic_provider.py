"""Anthropic LLM provider implementation."""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.llm.base import LLMProvider, LLMResult, LLMUsageStats

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    async def analyze_transcript(
        self, system_prompt: str, user_prompt: str
    ) -> LLMResult:
        result = await self._call_with_retry(system_prompt, user_prompt)
        content = result["content"]
        analysis = self._parse_response(content)
        usage = LLMUsageStats(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            latency_ms=result["latency_ms"],
            model=result["model"],
            provider="anthropic",
        )
        return LLMResult(analysis=analysis, usage=usage, raw_response=content)

    async def _raw_call(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text
        return {
            "content": content,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "model": response.model,
        }
