"""Pipeline: scan -> read -> count tokens -> keyword_score -> pack_batches -> call_batch -> map+schema-check -> reconcile -> assemble DocRecord.

All file I/O and metric updates live here; LLM boundary is strictly inside
`llm_service.call_batch`. Classifier is called for scoring + reconcile only.

Public entry point:
    run(config: dict, metrics: MetricsCollector, *, client=None, now=None,
        sleep=None) -> tuple[list[dict], dict]

Returns (records, run_metadata). file_store.write_files persists them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from classifier import LLMResult, StringMatch, keyword_score, reconcile
from llm_service import (
    BatchDoc,
    BatchFailure,
    BatchSuccess,
    LLMConfig,
    call_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _count_tokens(text: str, model: str) -> int:
    """tiktoken-based count with a graceful fallback to a cheap approximation
    if tiktoken / the encoding isn't available in the test env."""
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token. Only used if tiktoken absent.
        return max(1, len(text) // 4)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _maybe_inline(text: str, limit: int) -> str:
    """Return original text up to `limit` chars; else truncate with marker.
    See limit.md §1 — we inline by default and mark truncation, rather than
    externalising to a sidecar file, to keep output self-contained."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...truncated; original was {len(text)} chars]"


# ---------------------------------------------------------------------------
# Scan + read
# ---------------------------------------------------------------------------


class _ScannedDoc:
    __slots__ = (
        "source_path", "file_name", "content", "doc_id",
        "token_count", "error_reason",
    )

    def __init__(self, source_path: str, file_name: str):
        self.source_path = source_path
        self.file_name = file_name
        self.content: str = ""
        self.doc_id: str = ""
        self.token_count: int = 0
        self.error_reason: Optional[str] = None


def _scan(docs_dir: str) -> list[_ScannedDoc]:
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs_dir does not exist: {docs_dir}")
    out: list[_ScannedDoc] = []
    for root, dirs, files in os.walk(docs_dir):
        # Sort in place so scan order is stable across platforms/runs.
        dirs.sort()
        files.sort()
        for name in files:
            path = os.path.join(root, name)
            out.append(_ScannedDoc(source_path=path, file_name=name))
    return out


def _read_one(doc: _ScannedDoc, model: str, max_single_doc_tokens: int) -> None:
    """Populate doc.content / doc_id / token_count OR doc.error_reason.

    error_reason uses the snake_case codes from impl-spec §3 enum table.
    doc_id falls back to sha256(source_path) when content could not be read
    (non_txt_suffix / non_utf8_encoding); otherwise sha256(content).
    """
    if not doc.file_name.lower().endswith(".txt"):
        doc.error_reason = "non_txt_suffix"
        doc.doc_id = _doc_id(doc.source_path)
        return
    try:
        with open(doc.source_path, "r", encoding="utf-8") as f:
            doc.content = f.read()
    except UnicodeDecodeError:
        doc.error_reason = "non_utf8_encoding"
        doc.doc_id = _doc_id(doc.source_path)
        return
    doc.doc_id = _doc_id(doc.content)
    if doc.content == "":
        doc.error_reason = "empty_file"
        return
    doc.token_count = _count_tokens(doc.content, model)
    if doc.token_count > max_single_doc_tokens:
        doc.error_reason = "oversize"


# ---------------------------------------------------------------------------
# Batch packing (impl-spec §7)
# ---------------------------------------------------------------------------


def pack_batches(
    docs: list[_ScannedDoc],
    batching: dict,
) -> list[list[_ScannedDoc]]:
    """Token-budget packing. Only docs without error_reason participate."""
    ctx = int(batching.get("model_context_window", 128000))
    overhead = int(batching.get("prompt_overhead_tokens", 500))
    per_doc_out = int(batching.get("output_margin_per_doc_tokens", 300))

    batches: list[list[_ScannedDoc]] = []
    current: list[_ScannedDoc] = []
    current_tokens = 0
    for doc in docs:
        if doc.error_reason is not None:
            continue
        projected_output = per_doc_out * (len(current) + 1)
        available = ctx - overhead - projected_output
        if current and current_tokens + doc.token_count > available:
            batches.append(current)
            current = [doc]
            current_tokens = doc.token_count
        else:
            current.append(doc)
            current_tokens += doc.token_count
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# Assemble DocRecord
# ---------------------------------------------------------------------------


def _string_match_to_dict(sm: Optional[StringMatch]) -> Optional[dict]:
    if sm is None:
        return None
    return {"keyword": sm.keyword, "hits": sm.hits, "score": sm.score}


def _assemble_record(
    doc: _ScannedDoc,
    batch_id: Optional[str],
    sm: Optional[StringMatch],
    llm: Optional[LLMResult],
    extracted_fields: Optional[dict],
    reconciled,
    inline_text_max_chars: int,
) -> dict:
    return {
        "doc_id": doc.doc_id,
        "source_path": doc.source_path.replace("\\", "/"),
        "route": reconciled.route,
        "classification": {
            "string_match": _string_match_to_dict(sm),
            "llm": (
                {"keyword": llm.keyword, "score": llm.score}
                if llm is not None else None
            ),
            "conflict": reconciled.conflict,
            "final_confidence": reconciled.final_confidence,
            "review_reason": reconciled.review_reason,
        },
        "extracted_fields": extracted_fields,
        "original_text": _maybe_inline(doc.content, inline_text_max_chars),
        "batch_id": batch_id,
        "error_reason": doc.error_reason,
    }


def _assemble_unprocessed_record(doc: _ScannedDoc) -> dict:
    """Unified pre-LLM gate record shape (impl-spec §3 unprocessed_file).

    route=null, string_match=null, llm=null, conflict=false,
    final_confidence=0.0, review_reason=null, extracted_fields=null,
    original_text=null, batch_id=null. `error_reason` is a snake_case
    code from the §3 enum table, already set on `doc`.
    """
    return {
        "doc_id": doc.doc_id,
        "source_path": doc.source_path.replace("\\", "/"),
        "route": None,
        "classification": {
            "string_match": None,
            "llm": None,
            "conflict": False,
            "final_confidence": 0.0,
            "review_reason": None,
        },
        "extracted_fields": None,
        "original_text": None,
        "batch_id": None,
        "error_reason": doc.error_reason,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(
    config: dict,
    metrics: Any,
    *,
    client: Any = None,
    sleep: Optional[Callable[[float], None]] = None,
    now: Optional[Callable[[], str]] = None,
    call_batch_fn: Optional[Callable] = None,
) -> tuple[list[dict], dict]:
    """Run the pipeline.

    `client` / `call_batch_fn` are injected so tests can replace the LLM
    boundary. Production: app.py supplies an `openai.OpenAI()` client.
    Tests: supply `call_batch_fn=MagicMock(return_value=BatchSuccess(...))`.
    """
    t0 = time.perf_counter()
    now = now or _iso_now
    call_fn = call_batch_fn or call_batch
    sleep_fn = sleep or time.sleep

    processed_at = now()

    docs_dir = config["docs_dir"]
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    keywords: list[str] = config["keywords"]
    scoring_cfg: dict = config.get("keyword_scoring", {})
    reco_cfg: dict = config.get("reconciliation", {})
    schemas: dict[str, list[str]] = config["extraction_schemas"]
    batching_cfg: dict = config.get("batching", {})
    max_single = int(batching_cfg.get("max_single_doc_tokens", 50000))
    inline_limit = int(config.get("inline_text_max_chars", 10000))
    llm_cfg_dict = config.get("llm", {})
    llm_config = LLMConfig(
        model=model,
        max_retries=int(llm_cfg_dict.get("max_retries", 3)),
        retry_backoff_base_seconds=float(
            llm_cfg_dict.get("retry_backoff_base_seconds", 1.0)
        ),
        retry_on_status=tuple(llm_cfg_dict.get(
            "retry_on_status", (429, 500, 502, 503, 504))),
        json_mode=bool(llm_cfg_dict.get("json_mode", True)),
    )

    # ---- 1. scan + read ----
    scanned = _scan(docs_dir)
    for d in scanned:
        _read_one(d, model=model, max_single_doc_tokens=max_single)

    # Duplicate-content warning: same sha256 from different files (limit.md §11).
    # Both records are kept; the LLM will dedup by id so the second occurrence
    # hits the llm_response_missing path. Warned once per duplicate id.
    seen_ids: dict[str, str] = {}
    for d in scanned:
        if d.error_reason == "non_txt_suffix":
            continue
        if d.doc_id in seen_ids:
            logger.warning(
                "duplicate doc_id %s: %s and %s share content",
                d.doc_id, seen_ids[d.doc_id], d.source_path,
            )
        else:
            seen_ids[d.doc_id] = d.source_path

    # ---- 2. keyword score each ----
    # Pre-LLM gate files get string_match=None uniformly (impl-spec §6).
    # We do NOT run keyword_score on them even if content is readable
    # (e.g. oversize) — the record should not carry routing signal.
    sm_by_id: dict[str, Optional[StringMatch]] = {}
    for d in scanned:
        if d.error_reason is not None:
            sm_by_id[d.doc_id] = None
        else:
            sm_by_id[d.doc_id] = keyword_score(d.content, keywords, scoring_cfg)

    # ---- 3. pack batches (only healthy docs) ----
    batches = pack_batches(scanned, batching_cfg)
    metrics.total_batches += len(batches)
    # file_processed: docs that passed pre-LLM gate and entered a batch
    # (impl-spec §9). Counted here once per doc_id; decoupled from how many
    # output files the doc later lands in.
    metrics.file_processed += sum(len(b) for b in batches)

    # ---- 4. call LLM per batch, collect per-doc results ----
    llm_by_id: dict[str, Any] = {}                   # doc_id -> LLMDocResult
    batch_id_by_doc: dict[str, Optional[str]] = {}
    batch_failure_by_doc: dict[str, str] = {}        # doc_id -> failure kind enum

    for i, batch in enumerate(batches, start=1):
        batch_id = f"batch_{i:03d}"
        batch_docs = [BatchDoc(doc_id=d.doc_id, content=d.content) for d in batch]
        expected_ids = {d.doc_id for d in batch}
        for d in batch:
            batch_id_by_doc[d.doc_id] = batch_id

        outcome = call_fn(
            batch_docs, schemas, llm_config, client, metrics, sleep_fn
        )
        if isinstance(outcome, BatchFailure):
            # outcome.kind ∈ {"llm_api_error", "llm_envelope_error"}; strict enum
            # carried into DocRecord.error_reason per impl-spec §3.
            for doc_id in expected_ids:
                batch_failure_by_doc[doc_id] = outcome.kind
            continue
        assert isinstance(outcome, BatchSuccess)
        # Drop hallucinated ids (impl-spec §10 / A15): no metric; the paired
        # expected-id absence is what gets counted as llm_missing_docs below.
        for r in outcome.results:
            if r.doc_id not in expected_ids:
                logger.warning("LLM returned unknown doc_id %s, dropped", r.doc_id)
                continue
            llm_by_id[r.doc_id] = r

    # ---- 5. assemble per-doc records ----
    records: list[dict] = []
    for d in scanned:
        sm = sm_by_id.get(d.doc_id)
        batch_id = batch_id_by_doc.get(d.doc_id)

        # 5a. Pre-LLM gate (non_txt_suffix / non_utf8_encoding / empty_file /
        # oversize) → unified unprocessed_file shape. No kw_score, no reconcile,
        # no inline text. See impl-spec §6.
        if d.error_reason is not None:
            metrics.file_errors += 1
            records.append(_assemble_unprocessed_record(d))
            continue

        # 5b. Batch-level LLM failure → reconcile Path A (kw fallback).
        # batch_failure_by_doc[d.doc_id] is the strict enum kind from
        # llm_service (impl-spec §3): "llm_api_error" | "llm_envelope_error".
        if d.doc_id in batch_failure_by_doc:
            kind = batch_failure_by_doc[d.doc_id]
            d.error_reason = kind
            reconciled = reconcile(
                sm, None,
                schema_mismatch=False, missing_doc=False,
                batch_error_reason=kind,
                config=reco_cfg,
            )
            records.append(_assemble_record(
                d, batch_id, sm, None, None, reconciled, inline_limit,
            ))
            continue

        # 5c. LLM batch succeeded but this expected doc_id is absent from
        # results → count on the missing side (impl-spec §9 / A15).
        llm_res = llm_by_id.get(d.doc_id)
        if llm_res is None:
            d.error_reason = "llm_response_missing"
            metrics.llm_missing_docs += 1
            reconciled = reconcile(
                sm, None,
                schema_mismatch=False, missing_doc=True,
                batch_error_reason=None,
                config=reco_cfg,
            )
            records.append(_assemble_record(
                d, batch_id, sm, None, None, reconciled, inline_limit,
            ))
            continue

        # 5d. LLM returned — per-doc schema check.
        llm_kw = (llm_res.keyword or "").lower()
        extracted: Optional[dict] = llm_res.extracted_fields or {}

        if llm_kw not in schemas:
            # Unknown keyword: strict enum, log carries the bad value.
            logger.warning(
                "llm_schema_mismatch: unknown keyword %r for doc_id %s",
                llm_kw, d.doc_id,
            )
            d.error_reason = "llm_schema_mismatch"
            metrics.llm_schema_mismatch += 1
            reconciled = reconcile(
                sm, LLMResult(keyword=llm_kw, score=llm_res.score),
                schema_mismatch=True, missing_doc=False,
                batch_error_reason=None,
                config=reco_cfg,
            )
            # Unknown keyword cannot drive a valid route → degrade to "general"
            # (impl-spec §10). review_reason already = "llm_schema_mismatch".
            reconciled.route = "general"
            records.append(_assemble_record(
                d, batch_id, sm,
                LLMResult(keyword=llm_kw, score=llm_res.score),
                None, reconciled, inline_limit,
            ))
            continue

        llm_view = LLMResult(keyword=llm_kw, score=llm_res.score)
        expected_fields = schemas[llm_kw]
        missing = [f for f in expected_fields if f not in (extracted or {})]
        schema_mismatch = bool(missing)
        if schema_mismatch:
            logger.warning(
                "llm_schema_mismatch: missing fields %s for doc_id %s",
                missing, d.doc_id,
            )
            d.error_reason = "llm_schema_mismatch"
            metrics.llm_schema_mismatch += 1
            extracted = None

        reconciled = reconcile(
            sm, llm_view,
            schema_mismatch=schema_mismatch, missing_doc=False,
            batch_error_reason=None,
            config=reco_cfg,
        )
        records.append(_assemble_record(
            d, batch_id, sm, llm_view, extracted, reconciled, inline_limit,
        ))

    # ---- 6. run metadata ----
    ended_at = now()
    metrics.duration_seconds = round(time.perf_counter() - t0, 3)
    run_metadata = {
        "processed_at": processed_at,
        "ended_at": ended_at,
        "model": model,
        "metrics": metrics.as_dict() if hasattr(metrics, "as_dict") else _metrics_fallback(metrics),
    }
    return records, run_metadata


def _metrics_fallback(m: Any) -> dict:
    fields = (
        "file_processed", "file_errors", "total_batches",
        "llm_calls", "llm_retries", "llm_api_errors", "llm_tokens_used",
        "llm_schema_mismatch", "llm_missing_docs",
        "duration_seconds",
    )
    return {f: getattr(m, f, 0) for f in fields}
