# Implementation Specification

This document covers the engineering reasoning behind module boundaries, LLM failure handling, the reconcile branches, and the record schema. It is meant to be read alongside `README.md` (what/how) and `design/dataflow.md` (when/in what order).

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
| Any status in `config.llm.retry_on_status`, by default 429, 500, 502, 503, 504 | `llm_api_error` | yes, up to `max_retries` | `llm_retries++` per attempt; `llm_api_errors++` on exhaustion |
| Any other status | `llm_api_error` | no | `llm_api_errors++` |
| Network / timeout | `llm_api_error` | yes | same as retryable statuses |
| JSON parse fail, missing `results`, `score` out of range, `finish_reason == "length"` | `llm_envelope_error` | no | `llm_schema_mismatch` += batch size |

429 arrives from the SDK as `RateLimitError`, which subclasses `APIStatusError` and is therefore decided by the same status list as everything else, so `retry_on_status` is the only place retry policy is set. The wait between attempts is `retry_backoff_base_seconds * 2 ** attempt`, raised to the server's `Retry-After` when that header asks for longer and clamped at `retry_after_cap_seconds` so one throttled batch cannot stall the run.

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
| tiktoken count > `max_single_doc_tokens` | `oversize` |

Gate failures set `error_reason` but leave `review_reason` null. A pre-LLM rejection is an input-side problem — the operator fixes the file or the scan configuration — and does not need a human review queue. Consumers distinguish the two cases by `classification.llm is null AND route is null`.

### 3.4 Fixed `error_reason` enumeration

```
non_txt_suffix, non_utf8_encoding, empty_file, oversize,
llm_api_error, llm_envelope_error, llm_response_missing, llm_schema_mismatch
```

Where these codes stop short of what a production consumer would want, see [Trade-offs And Limitations](../README.md#trade-offs-and-limitations).

---

## 4. Reconcile — Five Named Paths

`classifier.reconcile` turns the `(string_match, llm_result)` pair into `(route, conflict, final_confidence, review_reason)`. The five reachable paths are named so tests and logs can refer to them unambiguously.

| Path | Condition | Route | Confidence | Conflict | Review |
|---|---|---|---|---|---|
| A | LLM unavailable (`BatchFailure`) | keyword label if present, else `general` | keyword score × fallback penalty, else 0.0 | false | mirrored from `error_reason` |
| B1 | keyword and LLM disagree, keyword comparably strong | LLM label | LLM score × conflict penalty | true | `conflict` |
| B2 | keyword and LLM disagree, LLM decisively stronger or keyword too weak | LLM label | LLM score | false | null |
| C | keyword agrees with LLM | LLM label | LLM score | false | null |
| D | no keyword hit, LLM valid | LLM label | LLM score | false | null |

"Comparably strong" is two conditions, both read from `config.reconciliation`: the raw keyword score reaches `conflict_keyword_min_score`, and the keyword score weighted by `string_match_conflict_weight` still reaches the LLM's own score. A full-score keyword hit on a secondary theme — "contract" appearing throughout a document whose actual subject is an urgent access failure — therefore does not drag a confident model call into human review.

Two design choices worth calling out:

- **B1 and B2 are not merged.** Both are disagreements, and both give the route to the model; they differ in what the operator is told. B1 raises the conflict flag and books the document for human review; B2 leaves the losing keyword visible in `classification.string_match` and lets the document through. Merging them would either flood the review queue with weak keyword noise or bury the genuine model-versus-dictionary conflicts among it.
- **B1 does not override the LLM's route.** The conflict flag is raised, the confidence is penalised, and the record goes to human review, but the LLM label still wins. The keyword scorer is a supplementary signal, not a voting peer. If the model is systematically wrong, that is a model-quality incident, not something to paper over with a keyword rule at inference time.

---

## 5. `review_reason` — Three Tiers

`review_reason` is a single enum field assigned by priority. Only one value can win, so the ordering is load-bearing.

0. **Batch delivery failure, mirrored from `error_reason`.** When the whole batch failed and no per-document result survives, `review_reason` takes the batch's value: `llm_api_error` or `llm_envelope_error`. Pre-LLM gate errors are excluded — an unreadable input file is the operator's problem, not a reviewer's, so its `review_reason` stays null.
1. **Per-document quality failure inside a delivered batch.** `llm_schema_mismatch` when the extracted fields do not satisfy the schema for the chosen keyword, `llm_missing_docs` when the model simply omitted the document. Schema mismatch is tested first.
2. **Classifier signal.** `conflict` (path B1) or `low_confidence` (`final_confidence` below `llm_low_confidence_threshold`). `conflict` wins when both apply.

Tier 0 outranks the rest because a confident keyword fallback on a failed batch can produce a high `final_confidence` on a document the model never saw. Judged on confidence alone, that document would land in `miscellaneous.json` unreviewed.

---

## 6. Data Structures

### 6.1 Pydantic at the LLM boundary

```python
class LLMDocResult(BaseModel):
    doc_id: str
    keyword: str           # not an enum — off-category values become per-doc schema mismatch
    score: float           # [0.0, 1.0]
    extracted_fields: dict # defaults to {}; the per-keyword schema is checked in pipeline

class LLMBatchResponse(BaseModel):
    results: list[LLMDocResult]
```

Token usage is read off the SDK response object rather than the parsed envelope, so it is not part of the model.

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
| `runtime_metadata.json` | counters, `input_file_ids` in processing order, run start/end timestamps, model name |

The first three are not mutually exclusive. A document that is both urgent and needs a human look appears in both `urgent.json` and `human_review.json`. Route priority and review risk are orthogonal questions; forcing a document into one bucket would hide one of the two answers.

---

## 7. Write Failure

A document that appears in both `urgent.json` and `human_review.json` is one delivery. When both files fail to write, a naive counter would double-count: `file_errors += 2` for one document, breaking `file_processed + file_errors == len(input_file_ids)`.

`file_store.write_files` tracks first-failure per `(doc_id, source_path)`. The first OSError against a given delivery key increments `file_errors` and decrements `file_processed`; subsequent failures for the same key are logged only. The invariant holds regardless of how many view files a single record participates in.

A failed view file does not abort the remaining writes, and the counters are snapshotted into `runtime_metadata.json` after all three view files have been attempted, so the metadata file reports the partial delivery rather than the state the pipeline ended on. `app.main` returns 1 when any output file failed, naming them in the log.
