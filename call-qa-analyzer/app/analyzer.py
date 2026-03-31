"""Core analysis logic — ties together prompts, LLM provider, and models."""

from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.llm.base import LLMProvider, LLMResult
from app.models import CallTranscriptRequest
from app.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


def get_provider() -> LLMProvider:
    """Factory: return the configured LLM provider instance."""
    provider_name = settings.llm_provider.lower()
    if provider_name == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    elif provider_name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. Use 'openai' or 'anthropic'."
        )


async def analyze_call(request: CallTranscriptRequest) -> LLMResult:
    """Analyze a single call transcript and return structured results + metadata."""
    provider = get_provider()

    system_prompt = build_system_prompt(request.department)
    user_prompt = build_user_prompt(
        call_id=request.call_id,
        agent_name=request.agent_name,
        call_date=request.call_date,
        call_duration_seconds=request.call_duration_seconds,
        department=request.department,
        transcript=request.transcript,
    )

    logger.info(
        "Analyzing call %s | agent=%s | dept=%s | duration=%ds",
        request.call_id,
        request.agent_name,
        request.department,
        request.call_duration_seconds,
    )

    result = await provider.analyze_transcript(system_prompt, user_prompt)

    # Ensure call_id matches the request (LLM might get it wrong)
    result.analysis.call_id = request.call_id

    logger.info(
        "Analysis complete for call %s | assessment=%s | escalation=%s | "
        "tokens=%d | latency=%.0fms | provider=%s | model=%s",
        request.call_id,
        result.analysis.overall_assessment.value,
        result.analysis.escalation_required,
        result.usage.total_tokens,
        result.usage.latency_ms,
        result.usage.provider,
        result.usage.model,
    )

    return result


async def analyze_batch(
    requests: list[CallTranscriptRequest],
) -> list[LLMResult]:
    """Analyze multiple transcripts concurrently."""
    tasks = [analyze_call(req) for req in requests]
    return await asyncio.gather(*tasks)
