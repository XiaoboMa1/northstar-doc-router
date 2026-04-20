# Implementation Specification

This document covers the engineering reasoning behind module boundaries, LLM failure handling, the reconcile branches, and the record schema. It is meant to be read alongside `README.md` (what/how) and `design/dataflow` (when/in what order).

---

## 1. Module Decoupling

| Module | Owns | Does not own |
|---|---|---|
| `app.py` | .env/config loading, required-field validation, client construction, top-level orchestration, exit code | any per-document logic |
| `pipeline.py` | recursive scan, pre-LLM gate, sha256 doc_id, tiktoken counting, batch packing, per-document result mapping, schema check, record assembly | HTTP calls, decision thresholds, JSON file writes |
| `llm_service.py` | prompt construction, OpenAI call, retry, JSON parsing, pydantic top-level shape check, `BatchFailure` construction | which documents to send, what the record looks like |
| `classifier.py` | `keyword_score`, `reconcile` — pure functions, no I/O | any notion of batches, files, or retries |
| `file_store.py` | three view files plus `runtime_metadata.json`, `OSError` handling, delivery-key dedup on write failure | deciding review/route status |
| `metrics.py` | `MetricsCollector` dataclass, one-line `show()` | increment policy (owned by whoever observes the event) |

This separation matters for five reasons:

1. **Failure mode locality.** An OpenAI error can only come from `llm_service.py`. A filter-rule change only touches `file_store.py`. A decision-threshold change only touches `classifier.py` and `config.json`. Bug reports are triaged to one file.
2. **Testability.** Each test file in `tests/` targets one module with mock inputs. `classifier.py` is pure-function and tested directly. `llm_service.py` is tested against a mock OpenAI client and canned payloads. `pipeline.py` is tested against a mock LLM service.
3. **Swappable LLM provider.** `llm_service.call_batch(batch, schemas)` returns either `list[LLMDocResult]` or a `BatchFailure`. A second provider is a second implementation behind that one function; `pipeline.py` does not change.
4. **Pure decision logic.** `classifier.py` is only a reconcile decision tree that can be exercised exhaustively with table-driven tests.
5. **Pure persistence logic.** `file_store.py` owns the three view files and the delivery-key dedup rule to simplify the main pipeline.

---

## 2. Control Flow

```
app.main
 └── pipeline.run
      ├── _scan_directory              (pre-LLM gate per file)
      │     └── classifier.keyword_score   (only if gate accepts)
      ├── _pack_batches                 (token budget)
      ├── for each batch:
      │     ├── llm_service.call_batch  (may return BatchFailure)
      │     ├── _map_results_by_doc_id  (drop unknown, mark missing)
      │     ├── _check_extraction_schema
      │     └── classifier.reconcile
      └── return (records, run_metadata)

app.main
 └── file_store.write_files             (three view files + metadata)
```

Exceptions are caught at three fixed places: inside `llm_service._invoke` (retry loop), inside `pipeline._scan_directory` (per-file gate rejections become records), and inside `file_store.write_files` (per-file OSError becomes a counter increment). Nothing else catches exceptions, which means any uncaught exception is a real bug, not a policy decision hidden in a `try/except`.

---

## 3. LLM Failure Handling

The LLM boundary is where the runtime spends most of its exceptional-path logic, because model calls fail in qualitatively different ways. Failures are grouped into three categories by who needs to act and what can still be trusted.

### 3.1 Whole-batch failure

Returned as `BatchFailure(kind=llm_api_error | llm_envelope_error)`. No per-document payload survives.

| Cause | Kind | Retry | Metric |
|---|---|---|---|
| 429, 500, 502, 503, 504 | `llm_api_error` | yes, `retry_max_attempts` | `llm_retries++` per attempt; `llm_api_errors++` on exhaustion |
| `RateLimitError` from SDK | `llm_api_error` | yes, unconditional | same as above |
| Network / timeout | `llm_api_error` | yes | same |
| JSON parse fail, missing `results`, `score` out of range, `finish_reason == "length"` | `llm_envelope_error` | no | `llm_schema_mismatch` += batch size |

When a `BatchFailure` comes back, the pipeline writes its `kind` into every document in that batch as `error_reason`, sets `classification.llm` to null, and mirrors `error_reason` into `review_reason` at the highest priority. This guarantees that no document whose model response is untrustworthy slips into `miscellaneous.json` unreviewed, regardless of how confident the keyword fallback looks.

### 3.2 Per-document failure inside a successful batch

The API returned a valid top-level shape, but one document's result is unusable. The rest of the batch ships normally.

| Cause | `error_reason` | `review_reason` | Metric |
|---|---|---|---|
| Expected `doc_id` not in results | `llm_response_missing` | `llm_missing_docs` | `llm_missing_docs++` |
| Returned `doc_id` not in expected list | (dropped with warning) | — | none |
| `keyword` not in `extraction_schemas`, or `extracted_fields` missing required keys | `llm_schema_mismatch` | `llm_schema_mismatch` | `llm_schema_mismatch++` |

Two design choices:

- **Counting only on the missing side.** If the model substitutes an invented ID for a real one, counting both would double-count one failure. Missing-side counting is the obligation the system failed to meet; unknown-side returns are just discarded noise.
- **The pydantic top-level shape check deliberately does not enforce the `keyword` enum.** An off-category value becomes a per-document schema mismatch rather than collapsing the whole batch. Enforcing the enum at the top level would let one bad item take down dozens of good ones.

### 3.3 Pre-LLM gate

The gate runs before any model call. Violations produce a unified unprocessed record and do not consume API quota.

| Cause | `error_reason` |
|---|---|
| Suffix is not `.txt` | `non_txt_suffix` |
| Bytes fail UTF-8 decode | `non_utf8_encoding` |
| Zero decoded characters | `empty_file` |
| tiktoken count > `max_doc_tokens` | `oversize` |

Gate failures set `error_reason` but leave `review_reason` null. A pre-LLM rejection is an input-side problem — the operator fixes the file or the scan configuration — and does not need a human review queue. Consumers distinguish the two cases by `classification.llm is null AND route is null`.

### 3.4 Fixed `error_reason` enumeration

```
non_txt_suffix, non_utf8_encoding, empty_file, oversize,
llm_api_error, llm_envelope_error, llm_response_missing, llm_schema_mismatch
```

see [](../README.md#trade-offs-and-limitations)

---

## 4. Reconcile — Five Named Paths

`classifier.reconcile` turns the `(string_match, llm_result)` pair into `(route, conflict, final_confidence, review_reason)`. The five reachable paths are named so tests and logs can refer to them unambiguously.

| Path | Condition | Route | Confidence | Conflict | Review |
|---|---|---|---|---|---|
| A | keyword agrees with LLM | LLM label | LLM score | false | null |
| B1 | LLM available, no keyword hit, non-urgent | LLM label | LLM score | false | null |
| B2 | LLM available, no keyword hit, urgent | `urgent` | LLM score | false | null |
| C | keyword vs LLM disagree, keyword strong enough | LLM label | LLM score × conflict penalty | true | `conflict` |
| D | LLM unavailable (BatchFailure) | keyword label if present, else `general` | keyword score × fallback penalty, else 0.0 | false | mirrored from `error_reason` |

Two design choices worth calling out:

- **B1 and B2 are not merged.** They look structurally identical but represent different operator scenarios: B1 is a normal non-urgent document with no keyword signal; B2 is an urgent document the keyword dictionary did not cover. Keeping them separate lets the keyword dictionary be audited by looking at B2-path records — if the same LLM label shows up repeatedly in B2 but not in A, the dictionary is missing a term.
- **Path C does not override the LLM's route.** The conflict flag is raised, the confidence is penalised, and the record goes to human review, but the LLM label wins. The keyword scorer is a supplementary signal, not a voting peer. If the model is systematically wrong, that is a model-quality incident, not something to paper over with a keyword rule at inference time.

---

## 5. `review_reason` — Three Levels

`review_reason` is a single enum field assigned by priority. Only one value can win, so the priority is load-bearing.

1. **Mirror from `error_reason`.** If `error_reason` is set by a whole-batch or per-document LLM failure, `review_reason` takes the same value (`llm_api_error`, `llm_envelope_error`, `llm_response_missing`, `llm_schema_mismatch`). Pre-LLM gate errors are excluded — they stay null.
2. **Per-document quality flag.** Already covered by level 1 via mirroring; listed separately so the ordering is explicit.
3. **Classifier signal.** `conflict` (path C) or `low_confidence` (final_confidence below `low_confidence_threshold`). `conflict` wins over `low_confidence` when both apply.

The priority ordering exists because a confident keyword fallback on a failed batch can produce a high `final_confidence` on a document the model never saw. Without level 1, that document would land in `miscellaneous.json` unreviewed.

---

## 6. Data Structures

### 6.1 Pydantic at the LLM boundary

```python
class LLMDocResult(BaseModel):
    doc_id: str
    keyword: str           # not an enum — off-category values become per-doc schema mismatch
    score: float           # [0.0, 1.0]
    extracted_fields: dict | None

class LLMBatchResponse(BaseModel):
    results: list[LLMDocResult]
    usage: dict | None
```

The top-level shape check is strict; the `keyword` value is not. Rationale in section 3.2.

### 6.2 `DocRecord` — the single record schema

```
doc_id, source_path, batch_id, original_text,
classification: {
    string_match: {keyword, score} | null,
    llm:          {keyword, score, extracted_fields} | null,
    conflict:     bool,
    route:        str | null,
    final_confidence: float,
    review_reason: enum | null,
    error_reason:  enum | null,
}
extracted_fields: dict | null   # null iff llm_schema_mismatch
```

`classification.llm` is null iff the LLM boundary could not return a trustworthy per-document result (whole-batch failure or missing document). `extracted_fields` is null iff the per-document schema check failed. These two fields are independent; a record can have a valid `classification.llm` block and still have `extracted_fields == null`.

### 6.3 Metrics

```
file_processed           successful records written
file_errors              pre-LLM gate rejections + output-write failures
llm_calls                successful batch deliveries
llm_retries              retry attempts (not final failures)
llm_api_errors           batches that exhausted retries
llm_schema_mismatch      envelope-check failures (×batch) + per-doc schema failures
llm_missing_docs         expected doc_ids absent from a delivered batch
llm_tokens_used          sum of API usage field across calls
```

`file_processed + file_errors == len(input_file_ids)`.

### 6.4 Output files

| File | Content |
|---|---|
| `miscellaneous.json` | records where `route != "urgent"` AND `review_reason is null` |
| `urgent.json` | records where `route == "urgent"` |
| `human_review.json` | records where `review_reason is not null` |
| `runtime_metadata.json` | counters + `input_file_ids` in processing order + config snapshot |

The first three are not mutually exclusive. A document that is both urgent and needs a human look appears in both `urgent.json` and `human_review.json`. Route priority and review risk are orthogonal questions; forcing a document into one bucket would hide one of the two answers.

---

## 7. Write Failure

A document that appears in both `urgent.json` and `human_review.json` is one delivery. When both files fail to write, a naive counter would double-count: `file_errors += 2` for one document, breaking `file_processed + file_errors == len(input_file_ids)`.

`file_store.write_files` tracks first-failure per `(doc_id, source_path)`. The first OSError against a given delivery key increments `file_errors` and decrements `file_processed`; subsequent failures for the same key are logged only. The invariant holds regardless of how many view files a single record participates in.
