"""Tests for llm_service.

All tests mock the OpenAI client; no real API calls. `sleep` is injected as a
no-op so retry tests run in microseconds.

Mapping to test-spec.md: T17 (retry success), T20 (markdown code fence),
T21 (pydantic envelope validation). Additional coverage for retry-exhausted,
non-retryable errors, malformed JSON, rate limit, empty batch.
"""

from __future__ import annotations

import json

import pytest

from llm_service import (
    BatchDoc,
    BatchFailure,
    BatchSuccess,
    LLMConfig,
    build_prompt,
    call_batch,
    strip_code_fence,
)

from tests.conftest import (
    make_api_status_error,
    make_chat_response,
    make_client,
    make_connection_error,
    make_rate_limit_error,
    make_timeout_error,
)


# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def docs() -> list[BatchDoc]:
    return [
        BatchDoc(doc_id="5246b82fc427784e", content="Invoice 48392 April sub."),
        BatchDoc(doc_id="21b94dc9f08e9427", content="Customer complaint text."),
    ]


@pytest.fixture
def schemas() -> dict[str, list[str]]:
    return {
        "invoice": ["invoice_number", "amount", "vendor", "date", "items"],
        "complaint": [
            "customer_id",
            "issue_category",
            "severity",
            "timeline",
            "requested_resolution",
        ],
        "general": ["topic", "action_items"],
    }


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4o-mini",
        max_retries=3,
        retry_backoff_base_seconds=0.0,  # doesn't matter, sleep is stubbed
        retry_on_status=(429, 500, 502, 503, 504),
        json_mode=True,
    )


@pytest.fixture
def no_sleep():
    """Substitute for time.sleep so retry tests don't wait."""
    return lambda _seconds: None


def _valid_response_for(docs):
    body = {
        "results": [
            {
                "doc_id": d.doc_id,
                "keyword": "invoice",
                "score": 0.9,
                "extracted_fields": {"invoice_number": "X"},
            }
            for d in docs
        ]
    }
    return make_chat_response(json.dumps(body), total_tokens=123)


# ===========================================================================
# Happy path
# ===========================================================================


def test_happy_path_returns_parsed_results(docs, schemas, config, metrics, no_sleep):
    client = make_client(_valid_response_for(docs))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert len(outcome.results) == len(docs)
    assert {r.doc_id for r in outcome.results} == {d.doc_id for d in docs}
    assert outcome.total_tokens == 123
    assert metrics.llm_calls == 1
    assert metrics.llm_retries == 0
    assert metrics.llm_api_errors == 0
    assert metrics.llm_schema_mismatch == 0
    assert metrics.llm_tokens_used == 123


def test_empty_docs_short_circuits_without_api_call(schemas, config, metrics, no_sleep):
    client = make_client(RuntimeError("must not call"))

    outcome = call_batch([], schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert outcome.results == []
    assert outcome.total_tokens == 0
    client.chat.completions.create.assert_not_called()
    assert metrics.llm_calls == 0


# ===========================================================================
# Retry behavior (T17)
# ===========================================================================


def test_retry_succeeds_after_two_503s(docs, schemas, config, metrics, no_sleep):
    # T17: attempt 1 -> 503, attempt 2 -> 503, attempt 3 -> success
    client = make_client(
        [
            make_api_status_error(503),
            make_api_status_error(503),
            _valid_response_for(docs),
        ]
    )

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert metrics.llm_retries == 2
    assert metrics.llm_api_errors == 0
    assert metrics.llm_calls == 1
    assert client.chat.completions.create.call_count == 3


def test_retry_exhausted_reports_failure(docs, schemas, config, metrics, no_sleep):
    # All calls fail with a retryable status; after max_retries we give up.
    client = make_client(
        [make_api_status_error(503) for _ in range(config.max_retries + 1)]
    )

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_api_error"
    assert "retries exhausted" in outcome.detail
    assert "503" in outcome.detail
    assert metrics.llm_api_errors == 1
    assert metrics.llm_retries == config.max_retries
    assert metrics.llm_calls == 0


def test_rate_limit_error_is_retryable(docs, schemas, config, metrics, no_sleep):
    client = make_client(
        [make_rate_limit_error(), _valid_response_for(docs)]
    )

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert metrics.llm_retries == 1
    assert metrics.llm_calls == 1


def test_connection_and_timeout_errors_are_retryable(
    docs, schemas, config, metrics, no_sleep
):
    client = make_client(
        [make_connection_error(), make_timeout_error(), _valid_response_for(docs)]
    )

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert metrics.llm_retries == 2


# ===========================================================================
# Non-retryable errors
# ===========================================================================


def test_non_retryable_400_fails_immediately(docs, schemas, config, metrics, no_sleep):
    client = make_client(make_api_status_error(400, "bad request"))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_api_error"
    assert "400" in outcome.detail
    assert "retries exhausted" not in outcome.detail  # distinguish from retry path
    assert metrics.llm_api_errors == 1
    assert metrics.llm_retries == 0
    assert client.chat.completions.create.call_count == 1


def test_non_retryable_401_fails_immediately(docs, schemas, config, metrics, no_sleep):
    client = make_client(make_api_status_error(401, "unauthorized"))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_api_error"
    assert "401" in outcome.detail
    assert metrics.llm_api_errors == 1


# ===========================================================================
# Parsing & pydantic validation
# ===========================================================================


def test_markdown_code_fence_is_stripped(docs, schemas, config, metrics, no_sleep):
    # T20: LLM wraps JSON in ```json ... ``` despite JSON mode — still parses.
    inner = json.dumps(
        {
            "results": [
                {
                    "doc_id": docs[0].doc_id,
                    "keyword": "invoice",
                    "score": 0.7,
                    "extracted_fields": {},
                },
                {
                    "doc_id": docs[1].doc_id,
                    "keyword": "complaint",
                    "score": 0.8,
                    "extracted_fields": {},
                },
            ]
        }
    )
    wrapped = f"```json\n{inner}\n```"
    client = make_client(make_chat_response(wrapped, total_tokens=42))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert len(outcome.results) == 2
    assert metrics.llm_calls == 1


def test_invalid_json_fails_batch_no_retry(docs, schemas, config, metrics, no_sleep):
    client = make_client(make_chat_response("not json at all"))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_envelope_error"
    assert "json" in outcome.detail.lower()
    assert metrics.llm_schema_mismatch == len(docs)
    assert metrics.llm_api_errors == 0
    assert metrics.llm_retries == 0
    assert client.chat.completions.create.call_count == 1


def test_pydantic_score_out_of_range_fails_batch(
    docs, schemas, config, metrics, no_sleep
):
    # T21: score=1.5 violates LLMDocResult.score constraint (ge=0, le=1).
    bad_body = json.dumps(
        {
            "results": [
                {
                    "doc_id": docs[0].doc_id,
                    "keyword": "invoice",
                    "score": 1.5,
                    "extracted_fields": {},
                }
            ]
        }
    )
    client = make_client(make_chat_response(bad_body))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_envelope_error"
    assert "envelope validation failed" in outcome.detail
    assert metrics.llm_schema_mismatch == len(docs)
    assert metrics.llm_api_errors == 0
    assert metrics.llm_retries == 0


def test_pydantic_missing_results_key_fails_batch(
    docs, schemas, config, metrics, no_sleep
):
    client = make_client(make_chat_response(json.dumps({"items": []})))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_envelope_error"
    assert "results" in outcome.detail.lower()
    assert metrics.llm_schema_mismatch == len(docs)


def test_invalid_keyword_not_rejected_at_llm_layer(
    docs, schemas, config, metrics, no_sleep
):
    """keyword is intentionally NOT constrained in the envelope model — this
    keeps bad keywords as a per-doc llm_schema_mismatch event in pipeline,
    rather than a whole-batch failure."""
    body = json.dumps(
        {
            "results": [
                {
                    "doc_id": docs[0].doc_id,
                    "keyword": "sPaM",  # mixed case, not in schemas
                    "score": 0.5,
                    "extracted_fields": {},
                },
                {
                    "doc_id": docs[1].doc_id,
                    "keyword": "complaint",
                    "score": 0.8,
                    "extracted_fields": {},
                },
            ]
        }
    )
    client = make_client(make_chat_response(body))

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert outcome.results[0].keyword == "sPaM"  # passed through as-is
    assert metrics.llm_calls == 1


# ===========================================================================
# Prompt construction
# ===========================================================================


def test_build_prompt_lists_expected_ids_and_schemas(docs, schemas):
    system, user = build_prompt(docs, schemas)

    for d in docs:
        assert d.doc_id in system
        assert f"[DOC_START id={d.doc_id}]" in user
        assert "[DOC_END]" in user
    for kw in schemas:
        assert kw in system


def test_build_prompt_allowed_list_is_sorted(schemas):
    system, _ = build_prompt([BatchDoc("id1", "x")], schemas)
    # sorted keys appear explicitly (stable across Python dict iteration order)
    assert "['complaint', 'general', 'invoice']" in system


def test_json_mode_passed_to_client(docs, schemas, config, metrics, no_sleep):
    client = make_client(_valid_response_for(docs))

    call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "gpt-4o-mini"


def test_json_mode_disabled_omits_response_format(
    docs, schemas, metrics, no_sleep
):
    cfg = LLMConfig(model="gpt-4o-mini", json_mode=False, max_retries=0)
    client = make_client(_valid_response_for(docs))

    call_batch(docs, schemas, cfg, client, metrics, sleep=no_sleep)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs


# ===========================================================================
# Helper unit tests
# ===========================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```json\n{}\n```", "{}"),
        ("```\n{}\n```", "{}"),
        ("  ```json\n{}\n```  ", "{}"),
        ("```json\n{\"a\":1}```", '{"a":1}'),  # no trailing newline
        ('{"a":1}', '{"a":1}'),  # no fence
        ("```\nline1\nline2\n```", "line1\nline2"),
    ],
)
def test_strip_code_fence(text, expected):
    assert strip_code_fence(text) == expected


# ===========================================================================
# Retry budget bookkeeping
# ===========================================================================


def test_max_retries_zero_means_one_attempt(docs, schemas, metrics, no_sleep):
    cfg = LLMConfig(model="gpt-4o-mini", max_retries=0, retry_backoff_base_seconds=0.0)
    client = make_client(make_api_status_error(503))

    outcome = call_batch(docs, schemas, cfg, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert metrics.llm_retries == 0
    assert client.chat.completions.create.call_count == 1


def test_backoff_uses_exponential_schedule(docs, schemas, config, metrics):
    """Verify sleep is called with base * 2**attempt for each retry."""
    cfg = LLMConfig(
        model="gpt-4o-mini",
        max_retries=3,
        retry_backoff_base_seconds=0.5,
    )
    calls: list[float] = []
    client = make_client(
        [
            make_api_status_error(503),
            make_api_status_error(503),
            make_api_status_error(503),
            _valid_response_for(docs),
        ]
    )

    call_batch(docs, schemas, cfg, client, metrics, sleep=calls.append)

    # attempt 0 -> 0.5, attempt 1 -> 1.0, attempt 2 -> 2.0
    assert calls == [0.5, 1.0, 2.0]


# ===========================================================================
# T26: finish_reason="length" → envelope schema_mismatch (truncated output)
# ===========================================================================


def _truncated_response(total_tokens: int = 50):
    """Build a response whose content is truncated JSON + finish_reason=length."""
    from types import SimpleNamespace
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"results": [{"doc_id": "aaa", "keywo'),
            finish_reason="length",
        )],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def test_finish_reason_length_yields_batch_schema_mismatch(
    docs, schemas, config, metrics, no_sleep
):
    client = make_client(_truncated_response())

    outcome = call_batch(docs, schemas, config, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchFailure)
    assert outcome.kind == "llm_envelope_error"
    # Distinct from generic "bad json" so triage can tell them apart.
    assert "length" in outcome.detail or "truncated" in outcome.detail
    # Envelope failure counts every doc in the batch.
    assert metrics.llm_schema_mismatch == len(docs)
    assert metrics.llm_api_errors == 0
    assert metrics.llm_retries == 0
    # No retry on envelope-layer failure.
    assert client.chat.completions.create.call_count == 1


# ===========================================================================
# T27: retry_on_status does NOT gate 429 (RateLimitError always retries)
# ===========================================================================


def test_retry_on_status_config_does_not_gate_429(
    docs, schemas, metrics, no_sleep
):
    """Documents limit.md §12: RateLimitError is always retryable, even if
    429 is absent from retry_on_status. If someone fixes _classify_exc to
    respect the config, this test flips."""
    cfg = LLMConfig(
        model="gpt-4o-mini",
        max_retries=3,
        retry_backoff_base_seconds=0.0,
        retry_on_status=(500, 502, 503, 504),  # deliberately exclude 429
        json_mode=True,
    )
    client = make_client([
        make_rate_limit_error(),
        _valid_response_for(docs),
    ])

    outcome = call_batch(docs, schemas, cfg, client, metrics, sleep=no_sleep)

    assert isinstance(outcome, BatchSuccess)
    assert metrics.llm_retries == 1
    assert metrics.llm_api_errors == 0
    assert metrics.llm_calls == 1


# ===========================================================================
# T28: cross-batch metrics isolation
# ===========================================================================


def test_cross_batch_metrics_isolation(docs, schemas, config, metrics, no_sleep):
    """Two sequential call_batch invocations sharing one MetricsStub.
    batch-1 succeeds (no retry), batch-2 exhausts retries. Ensure no state
    bleeds the other way."""
    # batch-1 side effects
    client1 = make_client(_valid_response_for(docs))
    outcome1 = call_batch(docs, schemas, config, client1, metrics, sleep=no_sleep)
    assert isinstance(outcome1, BatchSuccess)
    assert metrics.llm_calls == 1
    assert metrics.llm_retries == 0
    assert metrics.llm_api_errors == 0

    # batch-2: all attempts 503 → exhausted
    client2 = make_client([
        make_api_status_error(503) for _ in range(config.max_retries + 1)
    ])
    outcome2 = call_batch(docs, schemas, config, client2, metrics, sleep=no_sleep)
    assert isinstance(outcome2, BatchFailure)

    # Totals reflect both batches cleanly.
    assert metrics.llm_calls == 1                              # batch-1 only
    assert metrics.llm_api_errors == 1                         # batch-2 only
    assert metrics.llm_retries == config.max_retries           # batch-2 only
    assert metrics.llm_schema_mismatch == 0
