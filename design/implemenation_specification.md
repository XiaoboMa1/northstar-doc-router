# Implementation Specification

This document fixes the runtime contract and explains the engineering reasons behind it. It is the specification a second implementer should be able to read and reproduce behaviour from. Setup and consumption examples live in `README.md`; test coverage lives in `VERIFY.md`.

## 1. Key Assumptions

These are the assumptions that shape the module boundaries and the failure taxonomy. Other constraints inherited from the task brief are not repeated here.

**Every scanned file must produce exactly one record.** The invariant `file_processed + file_errors == len(input_file_ids)` depends on it. Without this assumption, `runtime_metadata.json` stops being a reliable audit surface: a consumer cannot confirm that a document they expected to see was actually handled.

**Keyword evidence and model confidence are different units.** The keyword score is a discrete lexical signal derived from counts and density; the LLM score is a continuous probability produced by the model. Treating them as interchangeable scalars and blending them would collapse decisions that should remain distinguishable (agreement, strong disagreement, weak disagreement). The reconciliation layer is therefore a decision tree, not a weighted average.

**Batch-level transport failures and per-document quality failures need different recovery paths.** A batch failure means nothing in the payload is trustworthy; a per-document failure means one record in an otherwise healthy batch is unusable. Collapsing them into a single error class would force the operator to reason about the whole batch every time the model misbehaves on one document.

**Downstream consumers prefer parser-stable fields over readable ones.** Error classification on the record is a fixed vocabulary; diagnostic narrative lives in logs and counters. This is a conscious trade between record readability and machine reliability — the machine side wins because records are written once and consumed many times.

## 2. Supported And Unsupported Files

Supported inputs are UTF-8 text files with a `.txt` suffix, non-empty after decoding, and within the per-document token limit. The directory is scanned recursively; subdirectory files are treated exactly like top-level files.

Four rejection classes produce records via the pre-LLM gate:

- A suffix that is not `.txt`.
- A byte sequence that `open(..., encoding="utf-8")` cannot decode.
- A file whose decoded content is the empty string.
- A file whose token count exceeds the configured single-document limit.

Rejected files emit a unified unprocessed record in which every decision-related field is neutralised: `route` is null (not the `general` label, because no semantic classification occurred), `string_match` and `llm` are null, `conflict` is false, `final_confidence` is zero, `review_reason` is null, and `extracted_fields`, `original_text`, and `batch_id` are null. Only `error_reason` carries the rejection class. A consumer identifies the unprocessed subset by `error_reason is not null AND classification.llm is null`.

Two behaviours in this area are intentional within the current scope but are the first candidates for refinement:

- An oversize document is rejected without running keyword scoring, even though the text was already read. A production variant might offer a keyword-fallback route for oversize files; this implementation chose the strict gate because mixing "rejected" and "partially routed" into one record shape increases consumer complexity.
- Whitespace-only content (`"   \n"`) passes the gate because "empty" is defined as zero characters after decode, not zero non-whitespace characters. The permissive definition lets the model decide whether the content is meaningful; the stricter definition would be an easy extension.

Non-goals in this scope: PDF, DOCX, HTML, and auto-detected encodings (UTF-16, Latin-1, GBK). Adding one would mean inserting a decoder layer before the gate; none of the downstream contracts would change.

## 3. Data Structures

### 3.1 Pydantic At The LLM Envelope

Pydantic is used at exactly one place: the envelope returned by the LLM. Two models:

```python
class LLMDocResult(BaseModel):
    doc_id: str
    keyword: str
    score: float = Field(ge=0.0, le=1.0)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)

class LLMBatchResponse(BaseModel):
    results: list[LLMDocResult]
```

`keyword` is deliberately a free string at this layer, not an enum. The envelope's job is to confirm that the transport delivered parseable JSON with the expected outer shape and numerically valid scores. Whether the model's returned label is a known category is a business concern, handled per-document in the pipeline. If the envelope enforced the enum, any off-category response would collapse the whole batch into a transport failure and penalise the healthy documents that happened to share the batch. The current split keeps the blast radius of a bad label limited to the document that produced it.

The pipeline performs two business-level checks on each result: that `keyword` is a key in `extraction_schemas`, and that every field listed under that schema is present in `extracted_fields` (the values may be null, but keys must exist).

### 3.2 Per-Document Record

Every output record carries the same shape: `doc_id`, `source_path`, `route`, `classification` (with `string_match`, `llm`, `conflict`, `final_confidence`, `review_reason`), `extracted_fields`, `original_text`, `batch_id`, and `error_reason`. The `classification` object keeps both the evidence (the two signals) and the outcome (the conflict flag, the confidence, the review reason) in the same block, so a reviewer can reproduce the decision from the record alone without correlating back to logs.

`extracted_fields` is tri-state by design: an object when the model succeeded and passed the schema check, and null for every other case — model unavailable, document missing from the response, schema violation. Collapsing the failure modes into one null value is a contract simplification; the specific reason is available via `error_reason`.

`final_confidence` is rounded to six decimal places at write time so golden comparisons remain stable without imposing rounding on in-memory math.

### 3.3 Metrics

`MetricsCollector` is a dataclass constructed once in the entry point and passed by reference into `pipeline` and `llm_service`. No module maintains a global counter; tests can construct a fresh instance per case without any teardown.

| Field | Meaning | Engineering purpose |
|---|---|---|
| `file_processed` | Documents that passed the pre-LLM gate and entered batching. | Decoupled from "records appearing in output files" so that overlap between views does not inflate the count. |
| `file_errors` | Pre-LLM gate rejections plus first-time per-record write failures. | A single operational signal for "did not deliver cleanly". |
| `total_batches` | Batches produced by the packer. | Comparing against `llm_calls` detects silent batch loss. |
| `llm_calls` | Successful API responses that passed envelope validation. | Cost attribution and batch health. |
| `llm_retries` | Individual retry attempts across the run. | Reliability signal separate from terminal failure count. |
| `llm_api_errors` | Batches where retries were exhausted or a non-retryable status was returned. | Alert threshold. |
| `llm_tokens_used` | `total_tokens` summed over successful batches only. | Cost attribution; failed batches have no usage field. |
| `llm_schema_mismatch` | Envelope failures (increment by batch size) plus per-document schema violations (increment by one). | One counter for "model response could not be consumed". |
| `llm_missing_docs` | Expected `doc_id` values absent from the model's response. | Obligation metric; counts on the missing side only. |
| `duration_seconds` | Wall clock from run to save. | Baseline for regression. |

The invariant `file_processed + file_errors == len(input_file_ids)` must hold for every run. Output write failures respect it through delivery-key deduplication: a record that appears in two failing output files counts once, not twice.

### 3.4 Output Files

| File | Filter | Purpose |
|---|---|---|
| `miscellaneous.json` | `route != "urgent" AND review_reason is null` | Normal-case bulk. Disjoint from the other two views. |
| `urgent.json` | `route == "urgent"` | Route-priority view for time-sensitive dispatch. |
| `human_review.json` | `review_reason is not null` | Risk view for human triage. |
| `runtime_metadata.json` | Single object | `processed_at`, `ended_at`, `model`, `input_file_ids`, `metrics`. |

`urgent.json` and `human_review.json` may share records; `miscellaneous.json` is the disjoint remainder. Overlap is intentional — route priority and review risk are orthogonal dimensions, and any exclusive partition would hide one of the two. A consumer that wants only the pure-conflict subset filters `human_review.json` on `classification.conflict == true`; no separate conflict file is emitted.

Keeping the three record views as bare arrays means the run metadata's schema can evolve (new metrics, new top-level fields) without invalidating record-level golden fixtures.

## 4. File Routing Logic

### 4.1 Keyword Scoring

`keyword_score` is a piecewise function over two independent gates: the hit count and the hit density. A document scores at the full level only when both gates pass, at a partial level when exactly one passes, at a weak level when neither passes but at least one hit exists, and zero otherwise. Ties between keywords are broken by their order in `config.keywords`, so the outcome is deterministic across Python runtimes. The purpose of the weak tier is to let the reconciliation layer distinguish "incidental lexical hit" from "real keyword signal" — without it, every non-zero match would read as strong evidence.

### 4.2 Reconciliation

Reconciliation is a decision tree with five reachable outcomes. Named by intent:

- **LLM unavailable.** When the LLM layer returned a batch failure, the record has no model signal. The route falls back to the keyword's label if one exists, or to `general` otherwise. Confidence is the keyword score multiplied by a fallback penalty (or zero if no keyword signal exists). No conflict is flagged, because there is no second signal to conflict with.
- **Signals agree.** The keyword and the LLM produce the same label. Route is the shared label; confidence is the LLM score unchanged.
- **No keyword signal with valid LLM.** There is no keyword match, but the LLM returned a result. Route and confidence come from the LLM.
- **Strong disagreement.** The keyword and the LLM produce different labels, and the keyword signal is strong enough — above the configured floor and comparable to the LLM score after rescaling — to warrant flagging. The route becomes the LLM's label, the conflict flag is raised, and the confidence is penalised. The record ends up in human review through the conflict reason.
- **Weak disagreement.** Labels differ but the keyword signal is too weak to challenge the LLM. The LLM wins silently; no conflict is flagged; confidence is the LLM score unchanged.

The rescaling step between the two signals before comparing them is why weak disagreement is distinct from strong disagreement. Without rescaling, any non-zero keyword match would compete on equal footing with the LLM, and the human review queue would fill with noise.

### 4.3 Review Reason Priority

The `review_reason` field is a single value chosen from the highest-priority condition that fires. The priority ordering, from most to least specific:

1. **Batch-level transport failure** (`llm_envelope_error`, `llm_api_error`). Mirrored from `error_reason` so a batch failure always reaches human review, regardless of how confident the keyword fallback looks.
2. **Per-document quality failure** (`llm_schema_mismatch`, `llm_missing_docs`). More specific than a generic uncertainty signal.
3. **Decision signals** (`conflict`, `low_confidence`). When both fire, `conflict` wins because a named disagreement is more actionable than a threshold alarm.

A pre-LLM rejection never produces a `review_reason`; it uses `error_reason` instead. Unprocessed records are not a human review concern — they are an input-quality concern, and the counters and file locations reflect that.

## 5. Error Handling Contract

### 5.1 Fixed Enum Vocabulary

`error_reason` takes one of eight values or null: `non_txt_suffix`, `non_utf8_encoding`, `empty_file`, `oversize`, `llm_api_error`, `llm_envelope_error`, `llm_response_missing`, `llm_schema_mismatch`. No dynamic suffix is ever appended. `review_reason` takes values from a related set: `conflict`, `low_confidence`, `llm_schema_mismatch`, `llm_missing_docs`, `llm_envelope_error`, `llm_api_error`, or null. Consumers filter with equality.

Diagnostic context — HTTP status, pydantic validation message, the specific missing field list, the hallucinated keyword string — is logged via the Python logger and counted in metrics. It is not embedded in the record, because embedding it would let the record schema drift over time and force consumers to parse strings.

### 5.2 Batch vs Per-Document Layering

A batch-level failure (exhausted retries, non-retryable status, envelope validation failure, `finish_reason == "length"` truncation) propagates one of two `error_reason` values to every expected document in that batch and sets `classification.llm` to null. A per-document failure leaves `classification.llm` populated and affects only the offending record. The two cases are visible from the record alone — null model block means batch failure; populated model block with `error_reason == "llm_schema_mismatch"` means per-document failure. A downstream system can branch on this without reading logs.
{Q. A downstream system can branch on this without reading logs. 什么是branch on? 让人误解，你说的是git branch吗？}

### 5.3 Retry Policy

Retryable statuses are hard-coded as `{429, 500, 502, 503, 504}` inside the LLM service. They are not exposed in configuration. The rationale is that the retryable set is the same information as the exception-classifier's branches; exposing it would introduce two sources of truth and allow them to drift. Backoff is exponential from a configurable base (`retry_backoff_base_seconds * 2 ** attempt`) for a configurable number of attempts (`max_retries`). The `Retry-After` header is ignored; this is a known gap acceptable in this scope.

### 5.4 Hallucinated Document IDs

If the model returns a result whose `doc_id` was never sent, the result is dropped with a warning. No counter is incremented. Counting lives on the missing side: for each expected `doc_id` absent from the response, `llm_missing_docs` advances by one. This is the only consistent way to measure obligation fulfilment without double-counting when the model substitutes a hallucinated ID for a real one.

### 5.5 Output Write Failures

`file_store` attempts to write all four output files independently. A per-file `OSError` does not abort the run — the remaining files are still attempted, so `runtime_metadata.json` always reaches disk when possible. For the failed file(s), records are deduplicated by `(doc_id, source_path)`; only the first failed delivery per key adjusts counters (`file_processed -= 1`, `file_errors += 1`), preserving the run-consistency invariant. If any file ultimately failed, the entry point exits with status 1 and prints the failed file names.
