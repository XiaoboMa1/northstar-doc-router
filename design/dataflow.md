# Dataflow

Runtime control flow and end-to-end examples for the normal path and the three distinct failure categories.

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant A as app.py
    participant P as pipeline.py
    participant C as classifier.py
    participant L as llm_service.py
    participant F as file_store.py

    A->>A: load .env and config.json, validate required fields
    A->>P: run(config, metrics)

    loop each file under docs_dir (recursive)
        P->>P: pre-LLM gate (suffix, utf-8 decode, non-empty, token count)
        alt gate rejects
            P->>P: build unified unprocessed record, set error_reason
        else gate accepts
            P->>C: keyword_score(text)
            C-->>P: string_match or None
        end
    end

    P->>P: pack_batches under token budget

    loop each batch
        P->>L: call_batch(batch, extraction_schemas)
        L->>L: build prompt with expected doc_ids
        L->>L: OpenAI call, retry on 429/500/502/503/504
        L->>L: parse JSON (tolerate markdown code fence)
        L->>L: pydantic top-level shape check
        alt batch delivered
            L-->>P: per-document results
            P->>P: map by doc_id, drop unknown ids, mark expected-missing
            P->>P: per-document schema check against extraction_schemas[keyword]
        else batch failed
            L-->>P: BatchFailure kind in {llm_api_error, llm_envelope_error}
        end
        P->>C: reconcile(string_match, llm_result, flags)
        C-->>P: route, conflict, final_confidence, review_reason
    end

    P-->>A: records, run_metadata
    A->>F: write_files(records, run_metadata)
    F->>F: mkdir output, split records into three list files
    F->>F: round final_confidence, write runtime_metadata.json
    F-->>A: success or per-file OSError
    A->>A: metrics.show(); exit 0 or exit 1
```

The run crosses three module boundaries: pipeline to classifier (pure decision), pipeline to LLM service (API call), and app to file store (persistence). Each boundary has one return shape — `string_match`, per-document results or a structured failure object, and the record list — and none of them lets exceptions leak across.

---

## Example 1 — Normal Path

Input: a short invoice document where the keyword `invoice` appears twice and the LLM returns the same label with 0.93 confidence.

1. Pre-LLM gate accepts the file.
2. `keyword_score` returns a match on `invoice` with a full-tier score because both the hit count and the density thresholds are met.
3. The document is packed into the first batch under the token budget.
4. `llm_service` calls the API once, receives valid JSON, passes the pydantic top-level shape check, and returns the per-document result.
5. The pipeline verifies that the `keyword` is known and that every field listed in `extraction_schemas["invoice"]` is present in the response (nullable values are acceptable; missing keys are not).
6. `reconcile` sees keyword and LLM agreeing; it takes the agreement branch. The route is `invoice`, the conflict flag is false, and `final_confidence` is the LLM's score unchanged. No review is flagged.
7. `file_store` writes the record into `miscellaneous.json` only. The document's `doc_id` is appended to `input_file_ids` in processing order.

Metric effects: `file_processed` increments by one, `llm_calls` by one, `llm_tokens_used` by the value reported in the API usage field.

---

## Example 2 — Pre-LLM Gate Rejection

Input: a `.png` file mixed into the directory, or a `.txt` file whose bytes fail UTF-8 decoding.

1. The gate catches the violation before the keyword scorer or the LLM runs.
2. No `keyword_score`, no batch packing, no model call. The pipeline assembles a unified unprocessed record: `route`, `string_match`, `llm`, `extracted_fields`, `original_text`, and `batch_id` are all null; `conflict` is false; `final_confidence` is 0.0; `review_reason` is null; `error_reason` is set to the specific enum (`non_txt_suffix`, `non_utf8_encoding`, `empty_file`, or `oversize`).
3. The record lands in `miscellaneous.json` only. Pre-LLM rejection uses `error_reason`, not `review_reason`, so it does not trigger the human review view.
4. `file_errors` increments by one; the LLM counters remain unchanged.

A consumer identifies pre-LLM rejections by `error_reason is not null AND classification.llm is null AND route is null`.

---

## Example 3 — Whole-Batch LLM Failure

Input: a batch of otherwise valid documents where either every retry attempt returns a 503, or the model returns a payload whose top-level JSON shape fails the pydantic check (missing `results` key, `score` out of range, or a `finish_reason == "length"` truncation).

1. `llm_service` exhausts retries or catches the validation failure and returns a structured `BatchFailure` with the kind set to `llm_api_error` or `llm_envelope_error`. The underlying exception does not escape the module.
2. The pipeline propagates that kind into every expected document in the batch as the document's `error_reason`. `classification.llm` becomes null for all of them.
3. `reconcile` runs the LLM-unavailable branch. If a keyword signal exists, the route becomes the keyword's label and `final_confidence` is the keyword score multiplied by the fallback penalty; otherwise the route is `general` with zero confidence.
4. `review_reason` is mirrored from `error_reason` at the highest priority. This guarantees every record in a failed batch reaches `human_review.json` regardless of how confident the keyword fallback looks.
5. Write-side filtering proceeds normally afterwards. A document whose fallback route is `urgent` appears in both `urgent.json` and `human_review.json`.

Metric effects for API exhaustion: `llm_retries` increments per attempt, `llm_api_errors` increments once for the batch. For shape-check failure: `llm_schema_mismatch` increments by the batch size. Error context — status codes, validation messages, truncation state — is logged; none of it is embedded in the record.

---

## Example 4 — Per-Document Quality Failure

Input: a batch where the API call succeeds and the top-level shape is valid, but one document's response is unusable.

Three sub-cases are handled distinctly:

**Missing document.** If an expected `doc_id` does not appear in the model's results, that document's `error_reason` is set to `llm_response_missing`. The keyword fallback determines its route. `review_reason` is set to `llm_missing_docs` and the `llm_missing_docs` counter advances by one. Other documents in the same batch are unaffected.

**Unknown document.** If the model returns a result with a `doc_id` not in the expected list, that result is dropped with a warning and no metric increment. Counting happens on the missing side, never on the unknown side, which avoids double-counting when the model substitutes an invented ID for a real one.

**Schema-violating document.** If the model returns a well-formed result for the right document but the `keyword` is not in `extraction_schemas`, or `extracted_fields` is missing required keys, `extracted_fields` is set to null and both `error_reason` and `review_reason` become `llm_schema_mismatch`. The record retains its `classification.llm` block — the decision context is still worth auditing, even though the structured extraction is discarded. The `llm_schema_mismatch` counter advances by one.

---

## Why Three Distinct Failure Categories

Because their operational implications differ.

Pre-LLM gate failures never reach the model. The operator's next action is to fix the input file or the scan configuration. Classification fields are null because no classification signal exists.

Whole-batch LLM failures are API-side incidents. The operator's next action is to investigate API quota, network, or model-side truncation. No per-document payload is trustworthy, so every record in the batch goes to human review automatically.

Per-document LLM failures are model-quality incidents inside an otherwise successful batch. The operator's next action is prompt adjustment or schema review. Only the affected records are flagged; the rest of the batch ships normally. Unknown and missing IDs are counted on one side only, so metrics reflect unmet obligations rather than noise volume.
