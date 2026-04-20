"""LLM service: OpenAI boundary for batch document classification.

Responsibilities (this module only):
    - Build the prompt from a batch of docs + extraction_schemas.
    - Call OpenAI chat.completions with JSON mode + exponential-backoff retry.
    - Parse the response, tolerating markdown code fences.
    - Validate the outer envelope with pydantic (LLMBatchResponse).
    - Return per-doc LLMDocResult list OR BatchFailure.

Non-responsibilities (owned by pipeline):
    - Mapping doc_ids back to source docs.
    - Validating extracted_fields against extraction_schemas[keyword].
    - Updating review_reasons / route / final_confidence.
    - Dropping hallucinated doc_idss.

Design notes:
    - The OpenAI client is dependency-injected (`client` arg) so tests can pass
      a Mock and the module never hits the network under test.
    - Pydantic validates only the envelope shape (doc_id, keyword string, score
      range, extracted_fields as dict). It intentionally does NOT constrain
      `keyword` to an enum — that check lives in the pipeline so an out-of-range
      keyword becomes a reviewable per-doc event ("llm_hallucination"), not a
      whole-batch failure.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models — LLM boundary envelope
# ---------------------------------------------------------------------------


class LLMDocResult(BaseModel):
    doc_id: str
    keyword: str
    score: float = Field(ge=0.0, le=1.0)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)


class LLMBatchResponse(BaseModel):
    results: list[LLMDocResult]


# ---------------------------------------------------------------------------
# Plain dataclasses — request / response shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchDoc:
    doc_id: str
    content: str


@dataclass(frozen=True)
class LLMConfig:
    model: str
    max_retries: int = 3
    retry_backoff_base_seconds: float = 1.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    json_mode: bool = True


@dataclass
class BatchSuccess:
    results: list[LLMDocResult]
    total_tokens: int


@dataclass
class BatchFailure:
    """Batch-level delivery failure.

    `kind` is the strict enum carried into DocRecord.error_reason
    (impl-spec §3): one of "llm_api_error" | "llm_envelope_error".
    `detail` is free-form context for logging only; it does NOT enter
    the record contract.
    """
    kind: str
    detail: str = ""

    @property
    def error(self) -> str:  # backwards-compat shim
        return self.kind


BatchOutcome = Union[BatchSuccess, BatchFailure]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_SYSTEM_TEMPLATE = (
    "You are a document classifier. For each input document, return a single "
    'JSON object of the form {{"results": [...]}} containing EXACTLY one entry '
    "per input doc_id.\n\n"
    "Each entry must have this shape:\n"
    "  doc_id: string (must match one of the input ids verbatim)\n"
    "  keyword: one of {allowed} (lowercase)\n"
    "  score: float in [0.0, 1.0] — your confidence\n"
    "  extracted_fields: object matching the schema for the chosen keyword; "
    "use null for missing values.\n\n"
    "Schemas per keyword:\n{schemas}\n\n"
    "You MUST and MUST ONLY return results for these doc_ids: {ids}"
)


def build_prompt(
    docs: list[BatchDoc],
    schemas: dict[str, list[str]],
) -> tuple[str, str]:
    """Return (system, user) messages for a batch."""
    allowed = sorted(schemas.keys())
    ids = [d.doc_id for d in docs]
    system = _SYSTEM_TEMPLATE.format(
        allowed=allowed,
        schemas=json.dumps(schemas, indent=2, sort_keys=True),
        ids=ids,
    )
    user = "\n".join(
        f"[DOC_START id={d.doc_id}]\n{d.content}\n[DOC_END]" for d in docs
    )
    return system, user


# ---------------------------------------------------------------------------
# Response cleanup
# ---------------------------------------------------------------------------


_CODE_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z",
    re.DOTALL,
)


def strip_code_fence(text: str) -> str:
    """Strip surrounding markdown code fence (```json ... ```) if present."""
    m = _CODE_FENCE_RE.match(text)
    return m.group(1) if m else text


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


class _Retryable(Exception):
    """Internal: API error eligible for retry."""


class _Fatal(Exception):
    """Internal: API error that should NOT retry."""


def call_batch(
    docs: list[BatchDoc],
    schemas: dict[str, list[str]],
    config: LLMConfig,
    client: Any,
    metrics: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> BatchOutcome:
    """Send a batch to the LLM. Returns BatchSuccess or BatchFailure.

    `client`: an object exposing `.chat.completions.create(**kwargs)` with the
    same contract as `openai.OpenAI()`. Inject a Mock in tests.

    `metrics`: optional duck-typed counter object with integer attributes
    `llm_calls`, `llm_retries`, `llm_api_errors`, `llm_schema_mismatch`,
    `llm_tokens_used`. Incremented in-place; pass `None` to disable.
    API-layer failure (retry exhausted / fatal) bumps `llm_api_errors`.
    Envelope pydantic validation failure bumps `llm_schema_mismatch` by
    `len(docs)` (whole batch treated as per-doc schema violations).

    `sleep`: injected for test determinism (tests pass a no-op).
    """
    if not docs:
        return BatchSuccess(results=[], total_tokens=0)

    system, user = build_prompt(docs, schemas)

    attempt = 0
    while True:
        try:
            response = _invoke(client, config, system, user)
        except _Retryable as e:
            if attempt >= config.max_retries:
                return _fail_api(
                    f"{e} (retries exhausted)", metrics
                )
            backoff = config.retry_backoff_base_seconds * (2 ** attempt)
            logger.warning(
                "LLM retry %d/%d after %.2fs: %s",
                attempt + 1,
                config.max_retries,
                backoff,
                e,
            )
            if metrics is not None:
                metrics.llm_retries += 1
            sleep(backoff)
            attempt += 1
            continue
        except _Fatal as e:
            return _fail_api(str(e), metrics)

        # Got a response; check finish_reason, parse + validate envelope.
        finish_reason = _finish_reason(response)
        content = _extract_content(response)
        if finish_reason == "length":
            return _fail_envelope(
                f"truncated output (finish_reason=length, {len(content)} chars)",
                metrics,
                batch_size=len(docs),
            )
        cleaned = strip_code_fence(content)
        try:
            parsed = LLMBatchResponse.model_validate_json(cleaned)
        except ValidationError as e:
            err = e.errors()[0]
            msg = f"envelope validation failed: {err.get('msg')} at {err.get('loc')}"
            return _fail_envelope(msg, metrics, batch_size=len(docs))
        except ValueError as e:
            # JSON decode errors (raised by pydantic for malformed JSON).
            return _fail_envelope(
                f"invalid json ({e})",
                metrics,
                batch_size=len(docs),
            )

        total_tokens = _tokens_of(response)
        if metrics is not None:
            metrics.llm_calls += 1
            metrics.llm_tokens_used += total_tokens
        return BatchSuccess(
            results=list(parsed.results),
            total_tokens=total_tokens,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _invoke(client: Any, config: LLMConfig, system: str, user: str) -> Any:
    """Call the chat completion API; classify errors as _Retryable / _Fatal."""
    # Imported lazily so this module is importable even when the optional
    # `openai` package is missing (e.g. in a minimal lint env). Tests that pass
    # a Mock client never hit the network.
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
    except ImportError:  # pragma: no cover
        APIConnectionError = APIStatusError = APITimeoutError = RateLimitError = ()  # type: ignore[assignment]

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        raise _Retryable(f"RateLimitError: {e}") from e
    except APIStatusError as e:
        status = getattr(e, "status_code", None)
        detail = getattr(e, "message", str(e))
        if status in config.retry_on_status:
            raise _Retryable(f"{status} {detail}") from e
        raise _Fatal(f"{status} {detail}") from e
    except (APIConnectionError, APITimeoutError) as e:
        raise _Retryable(f"{type(e).__name__}: {e}") from e
    except Exception as e:  # Mock side_effect, bad kwargs, etc.
        raise _Fatal(f"{type(e).__name__}: {e}") from e


def _extract_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        raise _Fatal(f"Malformed response: {e}") from e


def _finish_reason(response: Any) -> str | None:
    try:
        return getattr(response.choices[0], "finish_reason", None)
    except (AttributeError, IndexError):
        return None


def _tokens_of(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0)


def _fail_api(detail: str, metrics: Any) -> BatchFailure:
    logger.error("llm_api_error: %s", detail)
    if metrics is not None:
        metrics.llm_api_errors += 1
    return BatchFailure(kind="llm_api_error", detail=detail)


def _fail_envelope(detail: str, metrics: Any, batch_size: int) -> BatchFailure:
    """Envelope pydantic / JSON / finish_reason=length failure.
    Per impl-spec §10 + A21: metric bucket `llm_schema_mismatch` absorbs
    envelope failures (batch_size); error_reason enum stays distinct
    (`llm_envelope_error`)."""
    logger.error("llm_envelope_error: %s", detail)
    if metrics is not None:
        metrics.llm_schema_mismatch += batch_size
    return BatchFailure(kind="llm_envelope_error", detail=detail)
