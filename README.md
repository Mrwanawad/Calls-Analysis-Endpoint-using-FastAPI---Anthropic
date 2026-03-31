# Call QA Analyzer

AI-powered quality analysis for phone call transcripts at a pain management and neurology clinic. Replaces manual QA review (currently covering only 9% of calls) with automated, non-punitive analysis of 100% of calls.

## Quick Start

### 1. Install dependencies

```bash
cd call-qa-analyzer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

> **Default provider: Anthropic.** The app is pre-configured to use the **Anthropic API** (`claude-sonnet-4-20250514`) out of the box. An Anthropic API key is already set in `.env.example`. No OpenAI key is provided due to unavailability, so the app will only work with Anthropic by default.

#### Switching to OpenAI (or another provider)

Changing providers is straightforward — just update two values in your `.env` file:

```env
# Change this from "anthropic" to "openai"
LLM_PROVIDER=openai

# Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-key-here
```

No code changes are needed. The provider abstraction handles the rest automatically.

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test it

```bash
curl -X POST http://localhost:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d @sample_transcripts/clean_call.json
```

### 5. Run the evaluation script

```bash
# With the server running in another terminal:
python evaluate.py
```

---

## API Endpoints

### `POST /analyze-call`

Analyzes a single call transcript. Accepts a JSON body with `call_id`, `agent_name`, `call_date`, `call_duration_seconds`, `department`, and `transcript`. Returns structured quality analysis.

### `POST /batch-analyze`

Analyzes multiple transcripts concurrently. Accepts `{"transcripts": [...]}` and returns `{"results": [...]}`.

### `GET /health`

Returns `{"status": "ok", "provider": "<configured_provider>"}`.

---

## How It Works: Prompting Strategy

The prompting strategy is designed around a core principle: **in healthcare QA, a false positive is worse than a missed minor issue.** Incorrectly flagging an agent for a HIPAA violation they didn't commit damages morale (the exact problem this system replaces). Missing a minor phrasing issue is recoverable.

### System prompt structure

1. **Role framing**: The LLM is positioned as a "senior quality analyst" — not a critic. This framing produces more balanced assessments than adversarial prompts like "find all problems."

2. **Evidence-only rule**: The most repeated instruction is to only flag what is directly observable in the transcript text. This is reinforced multiple times because LLMs tend to infer issues that aren't there (e.g., assuming a short call means the agent was unhelpful).

3. **Severity calibration**: Explicit definitions for each severity level prevent the model from treating all issues as critical. "Escalate" is reserved for genuinely dangerous situations — HIPAA violations, overt hostility, or dangerous medical misinformation.

4. **Ambiguity handling**: When transcript text is unclear, the model is instructed to note the ambiguity rather than assume the worst. This matches the non-punitive philosophy.

5. **Department-specific rules**: Additional context is injected based on the call's department. For example:
   - **Scheduling**: Did the agent confirm date/time/location?
   - **Onboarding**: Was the lien agreement discussed?
   - **Records**: Was identity verified before disclosing records? (HIPAA-critical)

6. **Structured output enforcement**: The exact JSON schema is included in the prompt with field constraints. Combined with OpenAI's JSON mode (or careful parsing for Anthropic), this ensures reliable structured output.

### Temperature choice

Temperature is set to `0.1` — low enough for consistent, reproducible analysis, but not zero to avoid degenerate repetition patterns that can occur at `temperature=0`.

---

## Edge Case Handling

| Scenario | Approach |
|---|---|
| **Very short calls** (< 30s) | The prompt instructs the model to note brevity, score conservatively, and not penalize the agent. A 10-second wrong-number call should pass, not fail. |
| **Calls with no issues** | Return "pass" with at least one `positive_interaction` compliance flag. The model is told not to manufacture issues to seem thorough. |
| **Ambiguous transcript** | Note the ambiguity in the reasoning. Score accuracy conservatively (0.5–0.7) rather than guessing. |
| **Garbled/unclear segments** | Flag as unclear in the reasoning rather than interpreting garbled text. |
| **Transferred calls** | A transfer counts as resolution if it was appropriate (e.g., transferring to the correct department). |

---

## Architecture & Provider Abstraction

```
app/
├── main.py          # FastAPI endpoints + logging setup
├── models.py        # Pydantic input/output models (strict validation)
├── config.py        # Settings from .env via pydantic-settings
├── analyzer.py      # Core analysis orchestration
├── prompts.py       # Prompt templates + department-specific rules
└── llm/
    ├── base.py              # Abstract LLMProvider + retry logic
    ├── openai_provider.py   # OpenAI implementation
    └── anthropic_provider.py # Anthropic implementation
```

### Swapping providers

Change `LLM_PROVIDER` in `.env` from `openai` to `anthropic` (and set the appropriate API key). No code changes needed. The provider abstraction (`LLMProvider` base class) means adding a new provider requires implementing two methods: `analyze_transcript()` and `_raw_call()`.

### Retry logic

All LLM calls use `tenacity` with exponential backoff (3 attempts, 1–10s waits). This handles transient API failures without overwhelming the provider.

---

## Observability & Logging

All observability metrics are logged **server-side** in the terminal where `uvicorn` is running. They are **not** included in the API response — the endpoint returns only the structured JSON analysis. To view the metrics, check the terminal output while the server is processing requests.

### What gets logged

Every call to `/analyze-call` produces two log entries:

| Stage | Metrics logged |
|---|---|
| **Before LLM call** | `call_id`, `agent_name`, `department`, `call_duration_seconds` |
| **After LLM call** | `overall_assessment`, `escalation_required`, `total_tokens` (prompt + completion), `latency` (ms), `provider` (openai/anthropic), `model` (e.g. claude-sonnet-4-20250514) |
| **On failure** | Full exception message, retry attempt count |
| **Endpoint total** | `call_id`, total request time (ms) |

### Example terminal output

When you send a request, you will see something like this in your server terminal:

```
2025-01-15 10:30:45 | INFO     | app.analyzer | Analyzing call CALL-001 | agent=Maria Santos | dept=Scheduling | duration=245s
2025-01-15 10:30:47 | INFO     | app.analyzer | Analysis complete for call CALL-001 | assessment=pass | escalation=False | tokens=1523 | latency=2340ms | provider=anthropic | model=claude-sonnet-4-20250514
2025-01-15 10:30:47 | INFO     | app.main | POST /analyze-call completed | call_id=CALL-001 | total_time=2450ms
```

If an LLM call fails and retries, you will also see warning lines:

```
2025-01-15 10:30:45 | WARNING  | app.llm.base | LLM call failed (attempt 1), retrying: APIConnectionError(...)
```

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| **Single LLM call per transcript** | Simpler and faster than multi-step chains (e.g., separate calls for compliance, scoring, reasoning). Tradeoff: one prompt must handle all analysis dimensions. At this scope, one well-crafted prompt produces better results than chained calls that compound errors. |
| **JSON mode over function calling** | OpenAI's JSON mode is simpler to implement and works well for this use case. Function calling would add complexity without clear benefit for a single structured output. For Anthropic, we parse the JSON response directly. |
| **Low temperature (0.1)** | Sacrifices some nuance for consistency. In QA, reproducibility matters more than creativity — the same transcript should produce the same assessment. |
| **No caching** | Each call is analyzed fresh. For a production system, caching by transcript hash would reduce costs for duplicate calls. Omitted here to keep scope focused. |
| **Concurrent batch processing** | `asyncio.gather` sends all transcripts in parallel. This is fast but could hit rate limits on large batches. A production system would add concurrency limits or use a task queue. |
| **No database** | Per the spec — stateless endpoint. In production, results would be persisted for dashboards, trend analysis, and audit trails. |

---

## Sample Transcripts

| File | Scenario | Expected Result |
|---|---|---|
| `clean_call.json` | Professional scheduling call with identity verification, appointment confirmation, and pre-visit instructions | **pass** |
| `problematic_call.json` | Records call with HIPAA violations (no identity verification, reading diagnosis/medication aloud), rudeness, and improper records release | **escalate** |
| `edge_case_short_call.json` | 18-second wrong-number call — agent is polite, caller hangs up quickly | **pass** |
