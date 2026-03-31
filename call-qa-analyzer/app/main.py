"""FastAPI application — call transcript QA analysis endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.analyzer import analyze_batch, analyze_call
from app.config import settings
from app.models import (
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    CallAnalysisResponse,
    CallTranscriptRequest,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Call QA Analyzer",
    description=(
        "AI-powered quality analysis for phone call transcripts. "
        "Detects compliance issues, evaluates agent performance, "
        "and flags calls requiring escalation."
    ),
    version="1.0.0",
)

@app.get( '/' )
async def read_root():
    return { 'Message': 'Calls Analysis endpoint' }


@app.post("/analyze-call", response_model=CallAnalysisResponse)
async def post_analyze_call(request: CallTranscriptRequest) -> CallAnalysisResponse:
    """Analyze a single call transcript and return structured quality analysis."""
    start = time.perf_counter()
    try:
        result = await analyze_call(request)
    except Exception as exc:
        logger.error("Analysis failed for call %s: %s", request.call_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM analysis failed after retries: {exc}",
        )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /analyze-call completed | call_id=%s | total_time=%.0fms",
        request.call_id,
        total_ms,
    )
    return result.analysis


@app.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def post_batch_analyze(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze multiple call transcripts concurrently."""
    start = time.perf_counter()
    try:
        results = await analyze_batch(request.transcripts)
    except Exception as exc:
        logger.error("Batch analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM batch analysis failed: {exc}",
        )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /batch-analyze completed | count=%d | total_time=%.0fms",
        len(request.transcripts),
        total_ms,
    )
    return BatchAnalyzeResponse(results=[r.analysis for r in results])


@app.get("/health")
async def health_check() -> dict:
    """Simple health check."""
    return {"status": "ok", "provider": settings.llm_provider}
