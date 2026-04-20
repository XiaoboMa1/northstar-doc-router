# VERIFY

The suite focuses on control flow, metric accounting, and the output struture,  given the mock llm api.

```bash
python -m pytest ./tests       # 57 cases, runs in under a second
```

## Test File Structure And Responsibility

Each test file targets a single boundary of the system, and the file name identifies the module that regressed.

```
tests/
├── conftest.py             shared fixtures. The mock LLM service accepts
│                           pre-canned responses by doc_id, sequences of
│                           exceptions for retry tests, and a BatchFailure
│                           injector. Other fixtures provide temporary
│                           config/env wiring, a frozen clock for golden
│                           comparisons, and a sample-document builder.
├── test_pipeline.py        orchestration, pre-LLM gate, reconciliation
│                           outcomes, partition filters, write-failure
│                           accounting. Most of the suite lives here
│                           because most behaviour is orchestration.
├── test_classifier.py      pure-function checks on classifier.py.
├── test_llm_service.py     LLM boundary behaviour in isolation: parsing,
│                           retry, envelope validation, cross-batch
│                           metric isolation.
├── test_e2e.py             golden-file regression across all four outputs.
├── fixtures/               static model response payloads.
└── golden/                 canonical snapshots of the four output files.
```

## Case Summary By Pipeline Stage

Cases are grouped by the stage they exercise, following the order a document flows through the system.

### Startup And Configuration

Three cases verify that the entry point fails fast and never touches the input directory when configuration is incomplete: a missing required field in `config.json`, a missing `OPENAI_API_KEY`, and a missing `MODEL_NAME` each produce `exit(1)` with a diagnostic line on stderr and no output file created.

### Pre-LLM Input Gate

Five cases verify the input gate. Four are parameterised over the rejection reasons: empty file, non-`.txt` suffix, non-UTF-8 bytes, and token count above the single-document limit. Each produces a record with the unified unprocessed shape, no LLM call, and a `file_errors` increment. A fifth case runs a mixed input set that includes one valid document inside a subdirectory, verifying that the recursive walk reaches it and that counters distinguish processed documents from gated ones.

### Classifier Pure Function

One case verifies the tie-break rule in keyword scoring: when two keywords are equally strong, the winner is the one listed first in `config.keywords`.

### LLM Boundary

Six cases target the model boundary in isolation. One verifies that JSON wrapped in a markdown code fence is accepted after stripping the fence. One verifies that a retryable sequence (two 503 responses followed by success) increments the retry counter and returns a valid batch. One verifies that 429 is always treated as retryable by policy. Two verify envelope failure: a `score` outside `[0, 1]` and a `finish_reason == "length"` truncation both collapse the whole batch with `llm_schema_mismatch` incremented by the batch size, and every document in that batch inherits `error_reason == "llm_envelope_error"` and lands in `human_review.json`. The sixth verifies that metrics do not cross-contaminate between batches when one succeeds and another exhausts retries.

### Reconciliation Outcomes

Six cases cover the five reachable reconciliation paths plus their interaction with `review_reason`:

- Keyword and LLM agree: `conflict` is false, `review_reason` is null, record goes to `miscellaneous.json` only.
- No keyword hit, LLM valid with non-urgent label: record goes to `miscellaneous.json` only.
- No keyword hit, LLM labels the document urgent: record goes to `urgent.json` only with no review flag.
- Keyword and LLM disagree with the keyword signal strong enough to challenge the model: conflict flag is raised, LLM wins the route, confidence is penalised, `review_reason` is `"conflict"`, record goes to `human_review.json`.
- LLM API exhausted with a keyword signal present: keyword fallback provides the route, `classification.llm` is null, `error_reason` and `review_reason` both equal `"llm_api_error"`, record goes to `human_review.json`.
- Envelope failure with no keyword signal: double fallback to `route == "general"`, `final_confidence == 0.0`, `review_reason == "llm_envelope_error"`.

### Per-Document Quality Failures

Two cases cover the document-level failure modes that do not take down the whole batch. In the first, the LLM returns an unknown `doc_id` (hallucinated) and omits one expected `doc_id`; the hallucinated result is dropped without counting, and the missing document is marked with `error_reason == "llm_response_missing"`, `review_reason == "llm_missing_docs"`, and the `llm_missing_docs` counter advances by one. In the second, the LLM returns a well-formed result whose `extracted_fields` violates the per-category schema; the record retains its classification context, `extracted_fields` is set to `null`, and both `error_reason` and `review_reason` are set to `"llm_schema_mismatch"`.

### Output Partitioning And Write Failure

Three cases verify the persistence contract. The first checks the three filter rules and their disjointness/overlap relationships: `miscellaneous.json` is the disjoint remainder, `urgent.json` and `human_review.json` may overlap, and the three together cover every record. The second verifies that the pure-conflict subset is derivable by filtering `human_review.json` on `classification.conflict == true`, which is why no separate `conflict.json` is emitted. The third simulates an `OSError` on two output files that happen to share a record (a document that is both urgent and in human review). The deduplication logic in the write path ensures the record's delivery failure is counted once — one decrement of `file_processed`, one increment of `file_errors` — preserving the invariant `file_processed + file_errors == len(input_file_ids)`. The run exits with status 1 after logging the failed file names.

### End-To-End Regression

One case runs the full pipeline against the sample documents with a fixed clock and a canned model response set, then compares all four output files byte-for-byte against the committed golden snapshots. This is the regression net that catches any unintended change in the record schema, partition rule, metric, or ordering.

## Edge Case Handling

Several behaviours are worth flagging because naive implementations get them wrong:

**Counting hallucinated doc_ids.** Unknown returned IDs are dropped without metric increment; missing expected IDs are counted. If both sides counted, a model that hallucinated an ID to replace a missing one would be credited twice.

**Whole-batch vs per-document failure.** The pydantic envelope deliberately does not enforce the `keyword` enum. An off-category value from the model becomes a per-document schema mismatch, which keeps the rest of the batch usable; enforcing the enum at the envelope would collapse the whole batch over a single bad item.

**Batch failure visibility.** A batch-level LLM failure bypasses the normal confidence threshold by mirroring `error_reason` into `review_reason` at the highest priority. Without this, a confident keyword fallback could produce a high `final_confidence` on a document the model never actually saw, letting it land in `miscellaneous.json` unreviewed.

**Write-failure deduplication.** A record that appears in both `urgent.json` and `human_review.json` is one delivery. When both files fail to write, the logic tracks first-failure per `(doc_id, source_path)` so only one `file_errors` increment occurs. Without this, overlap between the two views would inflate the counter and break the consistency invariant any consumer might rely on.

**Whitespace-only content is not empty.** The gate rejects the empty string; it does not reject `"   \n"`. Whether this matters depends on how aggressively the downstream wants to pre-filter; the pipeline chooses the permissive definition so the model sees what the filesystem contains.
