"""End-to-end contract test (T16).

Runs the full pipeline against sample_docs/ with a stubbed call_batch_fn
that replays canned LLM responses from tests/fixtures/llm_responses.json.
Asserts current workflow/impl-spec contracts:

- output partitioning into miscellaneous / urgent / human_review
- runtime_metadata shape + metrics
- sample-doc specific route behavior (including doc_007 weak-kw B2 path)

Mock boundary = pipeline.run(call_batch_fn=...). No real OpenAI client.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pipeline
import file_store
from llm_service import BatchSuccess, LLMDocResult
from metrics import MetricsCollector

from tests.conftest import assert_full_run_metadata, write_config


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "llm_responses.json"
TOTAL_TOKENS = 560


def _copy_sample_docs(tmp_path: pathlib.Path) -> None:
    src = REPO_ROOT / "sample_docs"
    dst = tmp_path / "sample_docs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copy(p, dst / p.name)


def _normalise_paths(doc: dict, tmp_path: pathlib.Path) -> dict:
    prefix = str(tmp_path).replace("\\", "/").rstrip("/") + "/"
    if doc.get("source_path", "").startswith(prefix):
        doc["source_path"] = doc["source_path"][len(prefix):]
    return doc


def test_e2e_golden(tmp_path, monkeypatch):
    _copy_sample_docs(tmp_path)
    cfg = write_config(tmp_path)

    with open(FIXTURES, encoding="utf-8") as f:
        canned = json.load(f)

    def handler(docs, schemas, llm_cfg, client, metrics, sleep):
        metrics.llm_calls += 1
        metrics.llm_tokens_used += TOTAL_TOKENS
        results = []
        for d in docs:
            if d.doc_id not in canned:
                raise AssertionError(
                    f"sample doc {d.doc_id} missing from llm_responses.json fixture"
                )
            results.append(LLMDocResult(**canned[d.doc_id]))
        return BatchSuccess(results=results, total_tokens=TOTAL_TOKENS)

    # Deterministic duration + timestamps.
    ticks = iter([0.0, 2.34])
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(ticks))
    now_values = iter(["2026-04-18T10:23:45Z", "2026-04-18T10:23:47Z"])

    metrics = MetricsCollector()
    records, run_metadata = pipeline.run(
        cfg,
        metrics,
        client=None,
        sleep=lambda _s: None,
        now=lambda: next(now_values),
        call_batch_fn=handler,
    )
    file_store.write_files(records, run_metadata, cfg)

    with open(cfg["miscellaneous_file"], encoding="utf-8") as f:
        miscellaneous = json.load(f)
    with open(cfg["urgent_file"], encoding="utf-8") as f:
        urgent = json.load(f)
    with open(cfg["human_review_file"], encoding="utf-8") as f:
        human_review = json.load(f)
    with open(cfg["runtime_metadata_file"], encoding="utf-8") as f:
        runtime_metadata = json.load(f)

    for bucket in (miscellaneous, urgent, human_review):
        for doc in bucket:
            _normalise_paths(doc, tmp_path)

    # Partition contract checks.
    # doc_005_urgent_or_contract has string_match=contract (score 1.0) vs LLM=urgent
    # (0.9). reconcile path B: weighted_sm = 1.0 * 0.5 = 0.5 < 0.9, so B2 silent win
    # → route=urgent, conflict=False. It lands in urgent.json only.
    assert len(miscellaneous) == 6
    assert len(urgent) == 2
    assert len(human_review) == 0
    assert all(r["route"] != "urgent" for r in miscellaneous)
    assert all(r["classification"]["review_reason"] is None for r in miscellaneous)
    assert all(r["route"] == "urgent" for r in urgent)

    # Ordering: each output list sorted by source_path.
    with open(cfg["miscellaneous_file"], encoding="utf-8") as f:
        miscellaneous = json.load(f)
    with open(cfg["urgent_file"], encoding="utf-8") as f:
        urgent = json.load(f)
    for doc in miscellaneous + urgent:
        _normalise_paths(doc, tmp_path)
    assert [r["source_path"] for r in miscellaneous] == sorted(
        r["source_path"] for r in miscellaneous
    )
    assert [r["source_path"] for r in urgent] == sorted(
        r["source_path"] for r in urgent
    )

    # Scenario contract from workflow case 2B: doc_007 should be contract,
    # non-conflict (weak refund keyword match vs strong LLM contract).
    by_name = {r["source_path"].split("/")[-1]: r for r in (miscellaneous + urgent)}
    doc7 = by_name["doc_007_contract_with_stray_refund.txt"]
    assert doc7["route"] == "contract"
    assert doc7["classification"]["string_match"]["keyword"] == "refund"
    assert doc7["classification"]["string_match"]["score"] == 0.2
    assert doc7["classification"]["llm"]["keyword"] == "contract"
    assert doc7["classification"]["conflict"] is False
    assert doc7["classification"]["review_reason"] is None

    # Runtime metadata + metrics contract.
    expected_input_file_ids = [r["doc_id"] for r in records]
    assert_full_run_metadata(
        runtime_metadata,
        expected_input_file_ids=expected_input_file_ids,
        metrics_overrides={
            "file_processed": 8,
            "file_errors": 0,
            "total_batches": 1,
            "llm_calls": 1,
            "llm_retries": 0,
            "llm_api_errors": 0,
            "llm_tokens_used": TOTAL_TOKENS,
            "llm_schema_mismatch": 0,
            "llm_missing_docs": 0,
            "duration_seconds": 2.34,
        },
    )
