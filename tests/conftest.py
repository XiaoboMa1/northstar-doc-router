"""Shared test fixtures for the whole test suite.

- MetricsStub / openai-error factories / chat-response factory support llm_service
  unit tests.
- write_config / write_doc / make_call_batch_stub / llm_success / llm_failure
  support pipeline / classifier / e2e tests. All mock the LLM boundary at
  `llm_service.call_batch` via pipeline.run's `call_batch_fn` DI hook.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from llm_service import BatchFailure, BatchSuccess, LLMDocResult


# ---------------------------------------------------------------------------
# Metrics stub — duck-typed to match what llm_service expects.
# ---------------------------------------------------------------------------


@dataclass
class MetricsStub:
    file_processed: int = 0
    file_errors: int = 0
    total_batches: int = 0
    llm_calls: int = 0
    llm_retries: int = 0
    llm_api_errors: int = 0
    llm_tokens_used: int = 0
    llm_schema_mismatch: int = 0
    llm_missing_docs: int = 0
    duration_seconds: float = 0.0


@pytest.fixture
def metrics() -> MetricsStub:
    return MetricsStub()


# ---------------------------------------------------------------------------
# openai error factories
# ---------------------------------------------------------------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def make_api_status_error(status: int, message: str = "err") -> APIStatusError:
    """Build a real openai.APIStatusError with a controllable status_code."""
    response = httpx.Response(status_code=status, request=_request())
    return APIStatusError(message, response=response, body=None)


def make_rate_limit_error(message: str = "rate limited") -> RateLimitError:
    response = httpx.Response(status_code=429, request=_request())
    return RateLimitError(message, response=response, body=None)


def make_connection_error(message: str = "connection refused") -> APIConnectionError:
    return APIConnectionError(request=_request())


def make_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_request())


# ---------------------------------------------------------------------------
# Chat response factory — mimics openai.types.chat.ChatCompletion surface.
# ---------------------------------------------------------------------------


def make_chat_response(content: str, total_tokens: int = 100) -> SimpleNamespace:
    """Build a fake ChatCompletion-shaped object with .choices[0].message.content
    and .usage.total_tokens that llm_service reads.
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))
        ],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


# ---------------------------------------------------------------------------
# OpenAI-like client factory.
# ---------------------------------------------------------------------------


def make_client(side_effect: Any) -> MagicMock:
    """Build a MagicMock with .chat.completions.create configured from side_effect.

    `side_effect` can be a single response, a list of responses/exceptions, or
    an exception instance/class. Follows MagicMock conventions.
    """
    client = MagicMock()
    if isinstance(side_effect, list):
        client.chat.completions.create.side_effect = side_effect
    elif isinstance(side_effect, BaseException) or (
        isinstance(side_effect, type) and issubclass(side_effect, BaseException)
    ):
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = side_effect
    return client


# ---------------------------------------------------------------------------
# Pipeline / e2e helpers — production-shaped config + LLM boundary stubs.
# Mock boundary = llm_service.call_batch (via pipeline.run(call_batch_fn=...)).
# ---------------------------------------------------------------------------


DEFAULT_SAMPLE_CONFIG: dict = {
    "keywords": ["invoice", "complaint", "contract", "refund", "urgent"],
    "keyword_scoring": {
        "min_hits_for_full_score": 2,
        "density_threshold_for_full_score": 0.02,
        "partial_score": 0.5,
        "weak_score": 0.2,
    },
    "extraction_schemas": {
        "invoice": ["invoice_number", "amount", "vendor", "date", "items"],
        "contract": ["parties", "effective_date", "term", "obligations", "renewal_clause"],
        "complaint": ["customer_id", "issue_category", "severity", "timeline", "requested_resolution"],
        "refund": ["order_id", "amount", "reason", "original_payment_method"],
        "urgent": ["request_from", "issue", "requested_solution", "deadline"],
        "general": ["topic", "action_items"],
    },
    "reconciliation": {
        "conflict_keyword_min_score": 0.3,
        "string_match_conflict_weight": 0.5,
        "conflict_confidence_penalty": 0.8,
        "llm_fallback_confidence_penalty": 0.5,
        "llm_low_confidence_threshold": 0.5,
    },
    "batching": {
        "model_context_window": 128000,
        "prompt_overhead_tokens": 500,
        "output_margin_per_doc_tokens": 300,
        "max_single_doc_tokens": 50000,
    },
    "llm": {
        "max_retries": 3,
        "retry_backoff_base_seconds": 0.0,
        "retry_on_status": [429, 500, 502, 503, 504],
        "json_mode": True,
    },
    "inline_text_max_chars": 10000,
}


def write_config(tmp_path: pathlib.Path, **overrides: Any) -> dict:
    """Build a production-shaped config rebased under tmp_path.

    Files (`docs_dir`, `miscellaneous_file`, `urgent_file`,
    `human_review_file`, `runtime_metadata_file`) live under tmp_path. Any top-level key in
    `overrides` replaces the default; nested dicts must be passed whole.
    """
    import copy
    cfg = copy.deepcopy(DEFAULT_SAMPLE_CONFIG)
    cfg["docs_dir"] = str(tmp_path / "sample_docs")
    cfg["miscellaneous_file"] = str(tmp_path / "output" / "miscellaneous.json")
    cfg["urgent_file"] = str(tmp_path / "output" / "urgent.json")
    cfg["human_review_file"] = str(tmp_path / "output" / "human_review.json")
    cfg["runtime_metadata_file"] = str(tmp_path / "output" / "runtime_metadata.json")
    for k, v in overrides.items():
        cfg[k] = v
    (tmp_path / "sample_docs").mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# RunResult wrapper + T8-grade assertion helpers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Five-field bundle returned by tests/_run to avoid unpack-arity drift."""
    miscellaneous: list
    urgent: list
    human_review: list
    run_metadata: dict       # parsed runtime_metadata.json
    metrics: Any             # live MetricsCollector (in-process); exposes counters
                             # not serialised to disk (duration_seconds is; most others too).


_UNPROCESSED_SKELETON = {
    "route": None,
    "extracted_fields": None,
    "original_text": None,
    "batch_id": None,
    "_classification": {
        "string_match": None,
        "llm": None,
        "conflict": False,
        "final_confidence": 0.0,
        "review_reason": None,
    },
}


_METRICS_FIELDS = (
    "file_processed", "file_errors", "total_batches",
    "llm_calls", "llm_retries", "llm_api_errors", "llm_tokens_used",
    "llm_schema_mismatch", "llm_missing_docs", "duration_seconds",
)


def assert_full_doc_record(rec: dict, *, expected: dict) -> None:
    """Assert every DocRecord field. `expected` overrides a null-happy baseline.

    Baseline: batch_id=None, extracted_fields=None, original_text=None,
    classification.llm=None, classification.string_match=None,
    classification.conflict=False, classification.review_reason=None,
    error_reason=None. Tests only specify what differs.

    Special keys in `expected`:
      - `error_reason_prefix`: assert startswith instead of equality.
      - `error_reason_contains`: assert substring.
    """
    assert isinstance(rec, dict)
    # Top-level fields
    for k in ("doc_id", "source_path", "route", "classification",
              "extracted_fields", "original_text", "batch_id",
              "error_reason"):
        assert k in rec, f"DocRecord missing key: {k}"

    baseline = {
        "batch_id": None,
        "extracted_fields": None,
        "original_text": None,
        "error_reason": None,
    }
    for k, v in baseline.items():
        want = expected.get(k, v)
        assert rec[k] == want, f"{k}: expected {want!r}, got {rec[k]!r}"

    if "doc_id" in expected:
        assert rec["doc_id"] == expected["doc_id"]
    if "source_path" in expected:
        assert rec["source_path"] == expected["source_path"]
    if "route" in expected:
        assert rec["route"] == expected["route"]

    # Classification sub-object
    cls = rec["classification"]
    assert isinstance(cls, dict)
    cls_baseline = {
        "string_match": None,
        "llm": None,
        "conflict": False,
        "review_reason": None,
    }
    for k, v in cls_baseline.items():
        want = (expected.get("classification") or {}).get(k, v)
        assert cls.get(k) == want, f"classification.{k}: expected {want!r}, got {cls.get(k)!r}"
    if "final_confidence" in (expected.get("classification") or {}):
        assert cls.get("final_confidence") == expected["classification"]["final_confidence"]

    # error_reason prefix/contains helpers
    ur_prefix = expected.get("error_reason_prefix")
    ur_contains = expected.get("error_reason_contains")
    if ur_prefix is not None:
        assert rec["error_reason"] is not None
        assert rec["error_reason"].startswith(ur_prefix), \
            f"error_reason does not start with {ur_prefix!r}: {rec['error_reason']!r}"
    if ur_contains is not None:
        assert rec["error_reason"] is not None
        assert ur_contains in rec["error_reason"], \
            f"error_reason missing {ur_contains!r}: {rec['error_reason']!r}"


def assert_full_run_metadata(
    run_meta: dict,
    *,
    expected_input_file_ids: list,
    metrics_overrides: Optional[dict] = None,
) -> None:
    """Assert the full runtime_metadata.json envelope + every metric.

    Metrics baseline: every field in `_METRICS_FIELDS` defaults to 0 (or 0.0 for
    duration_seconds). `metrics_overrides` replaces specific fields. For
    duration_seconds, any non-negative float is accepted unless overridden.
    """
    assert isinstance(run_meta, dict)
    for k in ("processed_at", "ended_at", "model", "input_file_ids", "metrics"):
        assert k in run_meta, f"runtime_metadata missing key: {k}"
    assert isinstance(run_meta["processed_at"], str)
    assert isinstance(run_meta["ended_at"], str)
    assert isinstance(run_meta["model"], str) and run_meta["model"]
    assert run_meta["input_file_ids"] == expected_input_file_ids

    metrics = run_meta["metrics"]
    overrides = metrics_overrides or {}
    for f in _METRICS_FIELDS:
        if f == "duration_seconds":
            assert isinstance(metrics.get(f), (int, float))
            if f in overrides:
                assert metrics[f] == overrides[f]
            else:
                assert metrics[f] >= 0.0
        else:
            want = overrides.get(f, 0)
            assert metrics.get(f) == want, \
                f"metrics.{f}: expected {want}, got {metrics.get(f)}"


def assert_record_only_in(result: "RunResult", rec: dict, *, files: set) -> None:
    """Assert `rec` appears in each named file and NOT in the others.

    `files` ⊆ {"miscellaneous", "urgent", "human_review"}. runtime_metadata is
    handled separately via input_file_ids.
    """
    groups = {
        "miscellaneous": result.miscellaneous,
        "urgent": result.urgent,
        "human_review": result.human_review,
    }
    for name, bucket in groups.items():
        if name in files:
            assert rec in bucket, f"rec not in {name}"
        else:
            assert rec not in bucket, f"rec unexpectedly in {name}"


def handler_success_for(
    doc_id: str,
    keyword: str,
    score: float,
    extracted_fields: dict,
    *,
    total_tokens: int = 100,
) -> Callable:
    """Factory: produces a call_batch handler returning one doc's BatchSuccess.

    Use with `make_call_batch_stub(handler_success_for(...))` to keep tests
    readable while still exercising the real llm_service contract shape.
    """
    def _handler(docs, schemas, cfg, client, metrics, sleep):
        if metrics is not None:
            metrics.llm_calls += 1
            metrics.llm_tokens_used += total_tokens
        return BatchSuccess(
            results=[LLMDocResult(
                doc_id=doc_id,
                keyword=keyword,
                score=score,
                extracted_fields=extracted_fields,
            )],
            total_tokens=total_tokens,
        )
    return _handler


def write_doc(tmp_path: pathlib.Path, name: str, content: str, binary: bool = False) -> pathlib.Path:
    """Write a file under tmp_path/sample_docs/<name>."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / name
    if binary:
        path.write_bytes(content.encode("latin-1") if isinstance(content, str) else content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def llm_success(results: list[dict], total_tokens: int = 100) -> BatchSuccess:
    """Build a BatchSuccess from a list of dicts with LLMDocResult fields."""
    return BatchSuccess(
        results=[LLMDocResult(**r) for r in results],
        total_tokens=total_tokens,
    )


def llm_failure(error: str) -> BatchFailure:
    return BatchFailure(error=error)


def make_call_batch_stub(
    handler: Callable[[list, dict, Any, Any, Any, Any], Any]
) -> MagicMock:
    """Wrap a handler into a MagicMock matching call_batch's signature.

    handler(docs, schemas, cfg, client, metrics, sleep) -> BatchSuccess | BatchFailure.
    The returned MagicMock records every call so tests can use
    `stub.assert_not_called()` / `stub.call_count` / `stub.call_args_list`.
    """
    stub = MagicMock(side_effect=handler)
    return stub


def stub_never_called() -> MagicMock:
    """A call_batch stub that fails the test if invoked."""
    def _boom(*args, **kwargs):
        raise AssertionError("call_batch should not have been called")
    return MagicMock(side_effect=_boom)
