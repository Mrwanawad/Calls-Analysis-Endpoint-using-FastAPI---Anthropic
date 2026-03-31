#!/usr/bin/env python3
"""Evaluation script — runs sample transcripts against the API and validates results.

Usage:
    1. Start the server:  uvicorn app.main:app --port 8000
    2. Run this script:   python evaluate.py

The script:
  - Sends each sample transcript to POST /analyze-call
  - Validates the response matches the Pydantic output schema
  - Checks that the clean call passes and the problematic call is flagged
  - Prints a summary of results
"""

import asyncio
import json
import sys
from pathlib import Path
from colorama import Fore

import httpx

from app.models import CallAnalysisResponse

BASE_URL = "http://localhost:8000"
SAMPLES_DIR = Path(__file__).parent / "sample_transcripts"


# Expected outcomes for each sample transcript
EXPECTATIONS = {
    "CALL-001": {
        "file": "clean_call.json",
        "expected_assessment": "pass",
        "should_escalate": False,
        "description": "Clean scheduling call — should pass",
    },
    "CALL-002": {
        "file": "problematic_call.json",
        "expected_assessment": "escalate",
        "should_escalate": True,
        "description": "HIPAA violations & rudeness — should escalate",
    },
    "CALL-003": {
        "file": "edge_case_short_call.json",
        "expected_assessment": "pass",
        "should_escalate": False,
        "description": "Very short wrong-number call — should pass",
    },
}


async def run_evaluation() -> None:
    passed = 0
    failed = 0
    errors = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Verify server is running
        try:
            health = await client.get(f"{BASE_URL}/health")
            health.raise_for_status()
            print(f"Server health: {health.json()}\n")
        except httpx.ConnectError:
            print("ERROR: Cannot connect to server. Start it with:")
            print("  uvicorn app.main:app --port 8000")
            sys.exit(1)

        for call_id, spec in EXPECTATIONS.items():
            print(f"{'=' * 60}")
            print(f"Test: {spec['description']}")
            print(f"File: {spec['file']}")

            # Load sample
            sample_path = SAMPLES_DIR / spec["file"]
            with open(sample_path) as f:
                payload = json.load(f)

            # Send to API
            try:
                resp = await client.post(f"{BASE_URL}/analyze-call", json=payload)
                resp.raise_for_status()
            except Exception as exc:
                print(f"  FAIL: API request failed — {exc}")
                failed += 1
                errors.append(f"{call_id}: API error — {exc}")
                continue

            data = resp.json()

            # Validate schema
            try:
                analysis = CallAnalysisResponse.model_validate(data)
                print(f"  Schema validation: PASS")
            except Exception as exc:
                print(f"  Schema validation: FAIL — {exc}")
                failed += 1
                errors.append(f"{call_id}: Schema validation failed — {exc}")
                continue

            # Check assessment
            actual_assessment = analysis.overall_assessment.value
            expected_assessment = spec["expected_assessment"]
            assessment_ok = actual_assessment == expected_assessment
            print(
                f"  Assessment: {actual_assessment} "
                f"(expected: {expected_assessment}) — "
                f"{'PASS' if assessment_ok else 'FAIL'}"
            )

            # Check escalation
            escalation_ok = analysis.escalation_required == spec["should_escalate"]
            print(
                f"  Escalation: {analysis.escalation_required} "
                f"(expected: {spec['should_escalate']}) — "
                f"{'PASS' if escalation_ok else 'FAIL'}"
            )

            # Print details
            print(f"  Reasoning: {analysis.assessment_reasoning}")
            print(f"  Compliance flags: {len(analysis.compliance_flags)}")
            for flag in analysis.compliance_flags:
                print(f"    - [{flag.severity.value}] {flag.type.value}: {flag.description}")
            print(
                f"  Scores: professionalism={analysis.agent_performance.professionalism_score:.2f} "
                f"accuracy={analysis.agent_performance.accuracy_score:.2f} "
                f"resolution={analysis.agent_performance.resolution_score:.2f}"
            )

            if assessment_ok and escalation_ok:
                passed += 1
            else:
                failed += 1
                if not assessment_ok:
                    errors.append(
                        f"{call_id}: Expected assessment '{expected_assessment}', "
                        f"got '{actual_assessment}'"
                    )
                if not escalation_ok:
                    errors.append(
                        f"{call_id}: Expected escalation={spec['should_escalate']}, "
                        f"got {analysis.escalation_required}"
                    )

            print()

    # Summary
    print( Fore.CYAN + ("=" * 60))
    print( Fore.GREEN + f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  - {e}")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
