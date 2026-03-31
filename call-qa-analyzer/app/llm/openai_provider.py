"""OpenAI LLM provider implementation."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import LLMProvider, LLMResult, LLMUsageStats

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

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
            provider="openai",
        )
        return LLMResult(analysis=analysis, usage=usage, raw_response=content)

    async def _raw_call(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = response.choices[0].message
        usage = response.usage
        return {
            "content": message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "model": response.model,
        }
