"""Pipeline integration tests (T6–T8, T13, T18, T19, T22, T23, T24, T25).

Mock boundary = ``llm_service.call_batch`` via ``pipeline.run(call_batch_fn=...)``.
Assertion granularity = full DocRecord + full runtime_metadata + cross-file
destination + handler call-count (the T8 contract). Each test writes real
files with ``file_store.write_files`` and reloads four output files
(miscellaneous.json, urgent.json, human_review.json — bare lists — plus
runtime_metadata.json single object).

Pre-LLM gate cases (T6 empty, T7 non_txt, T8 oversize, T23 non_utf8) share
the unified unprocessed_file shape (impl-spec §6) and are covered by the
parameterized ``test_pre_llm_gate``. Other tests remain standalone.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pytest

import file_store
import pipeline
from metrics import MetricsCollector

from tests.conftest import (
    RunResult,
    assert_full_doc_record,
    assert_full_run_metadata,
    assert_record_only_in,
    llm_success,
    make_call_batch_stub,
    stub_never_called,
    write_config,
    write_doc,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _doc_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _path_doc_id(path) -> str:
    return hashlib.sha256(str(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def _run(cfg, stub) -> RunResult:
    metrics = MetricsCollector()
    records, meta = pipeline.run(
        cfg,
        metrics,
        client=None,
        sleep=lambda _s: None,
        now=lambda: "2026-04-18T12:00:00Z",
        call_batch_fn=stub,
    )
    file_store.write_files(records, meta, cfg)
    with open(cfg["miscellaneous_file"], encoding="utf-8") as f:
        miscellaneous = json.load(f)
    with open(cfg["urgent_file"], encoding="utf-8") as f:
        urgent = json.load(f)
    with open(cfg["human_review_file"], encoding="utf-8") as f:
        human_review = json.load(f)
    with open(cfg["runtime_metadata_file"], encoding="utf-8") as f:
        run_metadata = json.load(f)
    return RunResult(
        miscellaneous=miscellaneous,
        urgent=urgent,
        human_review=human_review,
        run_metadata=run_metadata,
        metrics=metrics,
    )


# ===========================================================================
# T6 / T7 / T8 / T23 — pre-LLM gate (parameterized)
# ===========================================================================


def _mk_empty(tmp_path):
    write_doc(tmp_path, "doc_empty.txt", "")
    return {
        "name": "doc_empty.txt",
        "expected_error_reason": "empty_file",
        "expected_doc_id": _doc_id(""),
    }


def _mk_non_txt(tmp_path):
    p = write_doc(tmp_path, "image.png", "\x89PNGbinarydata", binary=True)
    return {
        "name": "image.png",
        "expected_error_reason": "non_txt_suffix",
        "expected_doc_id": _path_doc_id(p),
    }


def _mk_non_utf8(tmp_path):
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    p = docs_dir / "bad_encoding.txt"
    p.write_bytes(b"\xff\xfe\x00\x01bad\x80\x81\x82")  # not valid UTF-8
    return {
        "name": "bad_encoding.txt",
        "expected_error_reason": "non_utf8_encoding",
        "expected_doc_id": _path_doc_id(p),
    }


def _mk_oversize(tmp_path, monkeypatch):
    # deterministic tokeniser: count chars
    monkeypatch.setattr(pipeline, "_count_tokens", lambda text, model: len(text))
    big_content = "invoice " * 10000  # 80000 chars > 50000 threshold
    write_doc(tmp_path, "big.txt", big_content)
    return {
        "name": "big.txt",
        "expected_error_reason": "oversize",
        "expected_doc_id": _doc_id(big_content),
    }


@pytest.mark.parametrize("case_id,builder", [
    ("T6_empty_file", _mk_empty),
    ("T7_non_txt_suffix", _mk_non_txt),
    ("T23_non_utf8_encoding", _mk_non_utf8),
])
def test_pre_llm_gate(tmp_path, case_id, builder):
    cfg = write_config(tmp_path)
    info = builder(tmp_path)
    stub = stub_never_called()
    result = _run(cfg, stub)

    stub.assert_not_called()
    assert len(result.miscellaneous) == 1
    rec = result.miscellaneous[0]

    assert_full_doc_record(rec, expected={
        "doc_id": info["expected_doc_id"],
        "route": None,
        "extracted_fields": None,
        "original_text": None,
        "batch_id": None,
        "error_reason": info["expected_error_reason"],
        "classification": {
            "string_match": None,
            "llm": None,
            "conflict": False,
            "final_confidence": 0.0,
            "review_reason": None,
        },
    })
    assert rec["source_path"].endswith(info["name"])
    assert_record_only_in(result, rec, files={"miscellaneous"})
    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[info["expected_doc_id"]],
        metrics_overrides={"file_errors": 1},
    )


def test_pre_llm_gate_oversize(tmp_path, monkeypatch):
    """T8 split out because it needs the monkeypatch fixture."""
    cfg = write_config(tmp_path)
    info = _mk_oversize(tmp_path, monkeypatch)
    stub = stub_never_called()
    result = _run(cfg, stub)

    stub.assert_not_called()
    assert len(result.miscellaneous) == 1
    rec = result.miscellaneous[0]

    assert_full_doc_record(rec, expected={
        "doc_id": info["expected_doc_id"],
        "route": None,
        "extracted_fields": None,
        "original_text": None,
        "batch_id": None,
        "error_reason": "oversize",
        "classification": {
            "string_match": None,
            "llm": None,
            "conflict": False,
            "final_confidence": 0.0,
            "review_reason": None,
        },
    })
    assert_record_only_in(result, rec, files={"miscellaneous"})
    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[info["expected_doc_id"]],
        metrics_overrides={"file_errors": 1},
    )


def test_files_are_created_under_tmp_path(tmp_path):
    """Sanity check for the confusion around file creation in tests.

    The files are real and written to pytest's per-test tmp_path; they are
    not created under the repository root.
    """
    cfg = write_config(tmp_path)
    p = write_doc(
        tmp_path,
        "visible_invoice.txt",
        "Invoice A-100\nPlease process this invoice today. Invoice attached.",
    )
    assert p.exists()

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 33
        return llm_success(
            [{
                "doc_id": docs[0].doc_id,
                "keyword": "invoice",
                "score": 0.9,
                "extracted_fields": {
                    "invoice_number": "A-100",
                    "amount": None,
                    "vendor": None,
                    "date": None,
                    "items": ["subscription"],
                },
            }],
            total_tokens=33,
        )

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    assert (tmp_path / "sample_docs" / "visible_invoice.txt").exists()
    assert (tmp_path / "output" / "miscellaneous.json").exists()
    assert (tmp_path / "output" / "urgent.json").exists()
    assert (tmp_path / "output" / "human_review.json").exists()
    assert (tmp_path / "output" / "runtime_metadata.json").exists()
    assert len(result.miscellaneous) == 1
    assert result.miscellaneous[0]["route"] == "invoice"
    assert stub.call_count == 1


def test_mixed_complex_sample_docs(tmp_path, monkeypatch):
    """Mixed corpus coverage: normal + pre-LLM gate failures in one run."""
    cfg = write_config(
        tmp_path,
        batching={
            "model_context_window": 128000,
            "prompt_overhead_tokens": 500,
            "output_margin_per_doc_tokens": 300,
            "max_single_doc_tokens": 100,
        },
    )
    monkeypatch.setattr(pipeline, "_count_tokens", lambda text, model: len(text))

    invoice_text = "Invoice 700. Please process this invoice immediately."
    general_text = "Weekly meeting notes. Action items tracked in board."
    oversize_text = "invoice " * 20  # 160 chars > max_single_doc_tokens=100

    write_doc(tmp_path, "invoice_ok.txt", invoice_text)
    write_doc(tmp_path, "general_ok.txt", general_text)
    write_doc(tmp_path, "empty.txt", "")
    write_doc(tmp_path, "blob.bin", "\\x89PNGbinarydata", binary=True)

    docs_dir = tmp_path / "sample_docs"
    bad_utf8 = docs_dir / "bad_encoding.txt"
    bad_utf8.write_bytes(b"\xff\xfe\x00\x01broken\x80")
    write_doc(tmp_path, "oversize.txt", oversize_text)

    expected_llm_ids = {_doc_id(invoice_text), _doc_id(general_text)}

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        assert {d.doc_id for d in docs} == expected_llm_ids
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 140
        out = []
        for d in docs:
            if d.doc_id == _doc_id(invoice_text):
                out.append({
                    "doc_id": d.doc_id,
                    "keyword": "invoice",
                    "score": 0.94,
                    "extracted_fields": {
                        "invoice_number": "700",
                        "amount": None,
                        "vendor": None,
                        "date": None,
                        "items": [],
                    },
                })
            else:
                out.append({
                    "doc_id": d.doc_id,
                    "keyword": "general",
                    "score": 0.83,
                    "extracted_fields": {
                        "topic": "meeting",
                        "action_items": ["track board"],
                    },
                })
        return llm_success(out, total_tokens=140)

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    assert len(result.miscellaneous) == 6
    assert result.urgent == []
    assert result.human_review == []

    by_name = {r["source_path"].rsplit("/", 1)[-1]: r for r in result.miscellaneous}
    assert by_name["invoice_ok.txt"]["route"] == "invoice"
    assert by_name["invoice_ok.txt"]["error_reason"] is None
    assert by_name["general_ok.txt"]["route"] == "general"
    assert by_name["general_ok.txt"]["error_reason"] is None

    assert by_name["empty.txt"]["error_reason"] == "empty_file"
    assert by_name["blob.bin"]["error_reason"] == "non_txt_suffix"
    assert by_name["bad_encoding.txt"]["error_reason"] == "non_utf8_encoding"
    assert by_name["oversize.txt"]["error_reason"] == "oversize"

    for name in ("empty.txt", "blob.bin", "bad_encoding.txt", "oversize.txt"):
        cls = by_name[name]["classification"]
        assert cls["string_match"] is None
        assert cls["llm"] is None
        assert cls["review_reason"] is None
        assert by_name[name]["route"] is None
        assert by_name[name]["extracted_fields"] is None
        assert by_name[name]["batch_id"] is None

    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[r["doc_id"] for r in result.miscellaneous],
        metrics_overrides={
            "file_processed": 2,
            "file_errors": 4,
            "total_batches": 1,
            "llm_calls": 1,
            "llm_tokens_used": 140,
        },
    )
    assert stub.call_count == 1


# ===========================================================================
# T13: LLM response missing a doc_id → llm_missing_docs
# ===========================================================================


def test_missing_doc_in_response(tmp_path):
    cfg = write_config(tmp_path)
    write_doc(tmp_path, "a.txt", "Invoice #1 from vendor alpha. Invoice total due.")
    write_doc(tmp_path, "b.txt", "Complaint about late delivery. Complaint escalated.")
    write_doc(tmp_path, "c.txt", "Contract renewal for party gamma. Contract term extended.")

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 120
        return llm_success(
            [
                {"doc_id": docs[0].doc_id, "keyword": "invoice", "score": 0.92,
                 "extracted_fields": {"invoice_number": "1", "amount": None,
                                      "vendor": "alpha", "date": None, "items": []}},
                {"doc_id": docs[2].doc_id, "keyword": "contract", "score": 0.88,
                 "extracted_fields": {"parties": ["gamma"], "effective_date": None,
                                      "term": None, "obligations": [],
                                      "renewal_clause": "extended"}},
            ],
            total_tokens=120,
        )

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    all_recs = result.miscellaneous + result.human_review
    by_name = {r["source_path"].rsplit("/", 1)[-1]: r for r in all_recs}
    a, b, c = by_name["a.txt"], by_name["b.txt"], by_name["c.txt"]

    assert a["route"] == "invoice"
    assert a["error_reason"] is None
    assert c["route"] == "contract"
    assert c["error_reason"] is None

    assert b["error_reason"] == "llm_response_missing"
    assert b["classification"]["llm"] is None
    assert b["classification"]["review_reason"] == "llm_missing_docs"
    assert b["extracted_fields"] is None
    assert b["classification"]["string_match"] is not None
    assert b["route"] == "complaint"
    assert_record_only_in(result, b, files={"human_review"})

    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[r["doc_id"] for r in [a, b, c] if r["doc_id"]],
        metrics_overrides={
            "file_processed": 3,
            "total_batches": 1,
            "llm_calls": 1,
            "llm_tokens_used": 120,
            "llm_missing_docs": 1,
        },
    )
    assert stub.call_count == 1


# ===========================================================================
# T18: fabricated doc_id → dropped + metric + warning
# ===========================================================================


def test_hallucinated_doc_id_dropped(tmp_path, caplog):
    cfg = write_config(tmp_path)
    write_doc(tmp_path, "a.txt", "Invoice number 9. Invoice amount 500.")
    write_doc(tmp_path, "b.txt", "Refund request. Refund for damaged goods.")

    fake_id = "f" * 16

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 90
        return llm_success(
            [
                {"doc_id": docs[0].doc_id, "keyword": "invoice", "score": 0.9,
                 "extracted_fields": {"invoice_number": "9", "amount": 500,
                                      "vendor": None, "date": None, "items": []}},
                {"doc_id": fake_id, "keyword": "invoice", "score": 0.7,
                 "extracted_fields": {"invoice_number": "x", "amount": 0,
                                      "vendor": None, "date": None, "items": []}},
            ],
            total_tokens=90,
        )

    stub = make_call_batch_stub(handler)
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        result = _run(cfg, stub)

    all_recs = result.miscellaneous + result.human_review
    ids = [r["doc_id"] for r in all_recs]
    assert fake_id not in ids

    by_name = {r["source_path"].rsplit("/", 1)[-1]: r for r in all_recs}
    a, b = by_name["a.txt"], by_name["b.txt"]
    assert a["route"] == "invoice"
    assert a["error_reason"] is None
    assert b["error_reason"] == "llm_response_missing"
    assert b["classification"]["review_reason"] == "llm_missing_docs"

    assert result.run_metadata["metrics"]["llm_missing_docs"] == 1
    assert any(fake_id in rec.message for rec in caplog.records)


# ===========================================================================
# T19: extracted_fields schema mismatch
# ===========================================================================


def test_extracted_fields_schema_mismatch(tmp_path):
    cfg = write_config(tmp_path)
    content = "Invoice #42 from vendor Acme. Invoice line items attached."
    write_doc(tmp_path, "inv.txt", content)

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 77
        return llm_success(
            [{
                "doc_id": docs[0].doc_id,
                "keyword": "invoice",
                "score": 0.91,
                "extracted_fields": {
                    "customer_id": "c1",
                    "issue_category": "billing",
                    "severity": "low",
                    "timeline": "today",
                    "requested_resolution": "refund",
                },
            }],
            total_tokens=77,
        )

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    assert len(result.human_review) == 1
    rec = result.human_review[0]
    assert rec["route"] == "invoice"
    assert rec["extracted_fields"] is None
    assert rec["error_reason"] == "llm_schema_mismatch"
    assert rec["classification"]["review_reason"] == "llm_schema_mismatch"
    assert result.run_metadata["metrics"]["llm_schema_mismatch"] == 1
    assert_record_only_in(result, rec, files={"human_review"})


# ===========================================================================
# T22: urgent as a first-class route
# ===========================================================================


def test_urgent_as_route(tmp_path):
    cfg = write_config(tmp_path)
    body = "Service cluster in region A is offline. Hot-fix required ASAP."
    write_doc(tmp_path, "ops.txt", body)

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        assert "urgent" in schemas
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 64
        return llm_success(
            [{
                "doc_id": docs[0].doc_id,
                "keyword": "urgent",
                "score": 0.9,
                "extracted_fields": {
                    "request_from": "ops",
                    "issue": "region-A cluster down",
                    "requested_solution": "hot-fix",
                    "deadline": "ASAP",
                },
            }],
            total_tokens=64,
        )

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    assert len(result.urgent) == 1
    rec = result.urgent[0]
    assert rec["route"] == "urgent"
    assert rec["classification"]["string_match"] is None
    assert rec["classification"]["llm"] == {"keyword": "urgent", "score": 0.9}
    assert rec["classification"]["conflict"] is False
    assert rec["classification"]["final_confidence"] == 0.9
    assert rec["classification"]["review_reason"] is None
    assert rec["error_reason"] is None
    assert_record_only_in(result, rec, files={"urgent"})

    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[rec["doc_id"]],
        metrics_overrides={
            "file_processed": 1,
            "total_batches": 1,
            "llm_calls": 1,
            "llm_tokens_used": 64,
        },
    )


# ===========================================================================
# T24: subdirectory file is scanned recursively
# ===========================================================================


def test_subdirectory_file_scanned(tmp_path):
    cfg = write_config(tmp_path)
    subdir = tmp_path / "sample_docs" / "subdir"
    subdir.mkdir(parents=True, exist_ok=True)
    nested = subdir / "nested.txt"
    nested_body = "Invoice in nested folder. Please process this invoice."
    nested.write_text(nested_body, encoding="utf-8")

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 41
        return llm_success(
            [{
                "doc_id": docs[0].doc_id,
                "keyword": "invoice",
                "score": 0.9,
                "extracted_fields": {
                    "invoice_number": None,
                    "amount": None,
                    "vendor": None,
                    "date": None,
                    "items": [],
                },
            }],
            total_tokens=41,
        )

    stub = make_call_batch_stub(handler)
    result = _run(cfg, stub)

    assert stub.call_count == 1
    assert len(result.miscellaneous) == 1
    rec = result.miscellaneous[0]
    assert rec["route"] == "invoice"
    assert rec["source_path"].endswith("subdir/nested.txt")
    assert rec["error_reason"] is None
    assert result.urgent == []
    assert result.human_review == []
    assert_full_run_metadata(
        result.run_metadata,
        expected_input_file_ids=[rec["doc_id"]],
        metrics_overrides={
            "file_processed": 1,
            "file_errors": 0,
            "total_batches": 1,
            "llm_calls": 1,
            "llm_tokens_used": 41,
        },
    )


# ===========================================================================
# T25: duplicate content (same sha256, different file names)
# ===========================================================================


def test_duplicate_content(tmp_path, caplog):
    cfg = write_config(tmp_path)
    content = "Invoice 99999 please review. Invoice charged."
    write_doc(tmp_path, "dup_a.txt", content)
    write_doc(tmp_path, "dup_b.txt", content)
    shared_id = _doc_id(content)

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += 50
        # LLM sees only one unique id (second overwrites first by id).
        return llm_success(
            [{
                "doc_id": shared_id,
                "keyword": "invoice",
                "score": 0.9,
                "extracted_fields": {"invoice_number": "99999", "amount": None,
                                     "vendor": None, "date": None, "items": []},
            }],
            total_tokens=50,
        )

    stub = make_call_batch_stub(handler)
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        result = _run(cfg, stub)

    # Two records produced, same doc_id, different source_path. Because the
    # LLM result map is keyed by doc_id, both records read the single result
    # back — so both land in the normal path. The warning is the user-visible
    # signal; the duplicate itself is not an error.
    all_recs = result.miscellaneous + result.urgent + result.human_review
    assert len(all_recs) == 2
    assert {r["doc_id"] for r in all_recs} == {shared_id}
    assert len({r["source_path"] for r in all_recs}) == 2

    reasons = [r["error_reason"] for r in all_recs]
    assert reasons == [None, None]

    assert any("duplicate doc_id" in rec.message for rec in caplog.records)

    # input_file_ids preserves duplicates (impl-spec §2.1).
    assert result.run_metadata["input_file_ids"] == [shared_id, shared_id]
    assert result.run_metadata["metrics"]["file_processed"] == 2
