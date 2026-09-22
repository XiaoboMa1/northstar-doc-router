# VERIFY

The suite focuses on control flow, metric accounting, and the output structure, with the LLM boundary mocked. No case needs an API key or a network.

```bash
python -m pytest ./tests       # 61 cases, runs in under a second
```

## Test File Structure And Responsibility

Each test file targets a single boundary of the system, and the file name identifies the module that regressed.

```
tests/
├── conftest.py             shared fixtures. The mock LLM service accepts
│                           pre-canned responses by doc_id, sequences of
│                           exceptions for retry tests, and a BatchFailure
│                           injector. Other fixtures build a production-shaped
│                           config under the test's own directory, a
│                           sample-document builder, and the full-record and
│                           full-metadata assertion helpers.
├── test_pipeline.py        orchestration, pre-LLM gate, reconciliation
│                           outcomes, partition filters. Most of the suite
│                           lives here because most behaviour is
│                           orchestration.
├── test_classifier.py      pure-function checks on classifier.py.
├── test_llm_service.py     LLM boundary behaviour in isolation: parsing,
│                           retry, top-level JSON shape check, cross-batch
│                           metric isolation.
├── test_e2e.py             full-pipeline regression across all four outputs.
└── fixtures/               static model response payloads.
```

## Case Summary By Pipeline Stage

Cases are grouped by the stage they exercise, following the order a document flows through the system.

### Pre-LLM Input Gate

Five cases verify the input gate. Four cover the rejection reasons — empty file, non-`.txt` suffix, non-UTF-8 bytes, and token count above the single-document limit — each producing a record with the unified unprocessed shape, no LLM call, and a `file_errors` increment. A fifth case runs a mixed input set that includes one valid document inside a subdirectory, verifying that the recursive walk reaches it and that counters distinguish processed documents from gated ones.

### Keyword Scoring

Three cases cover `classifier.keyword_score` directly: a document with no keyword at all returns `None` rather than a zero-scored match, a single hit lands on the partial tier instead of full score, and when two keywords are equally strong the winner is the one listed first in `config.keywords`.

### LLM Boundary

Thirty-two cases target the model boundary in isolation.

Parsing: JSON wrapped in a markdown code fence is accepted after the fence is stripped, checked over six input shapes. Two cases verify envelope failure — a `score` outside `[0, 1]` and a `finish_reason == "length"` truncation both collapse the whole batch with `llm_schema_mismatch` incremented by the batch size, and every document in that batch inherits `error_reason == "llm_envelope_error"` and lands in `human_review.json`. One case verifies the deliberate gap in that strictness: an off-category `keyword` passes the envelope check, because it is a per-document problem and must not take the batch down with it.

Retry policy: two 503s followed by success increment the retry counter and return a valid batch; an exhausted budget returns `llm_api_error`; connection and timeout errors retry while 400 and 401 fail on the first attempt; `max_retries=0` means exactly one attempt. A 429 is retried when `retry_on_status` lists it and fails immediately when it does not; the SDK reports 429 as `RateLimitError`, and the paired cases hold it to the configured list. Four cases cover the wait itself: the exponential schedule, a `Retry-After` header raising that wait but never shortening it, the configured cap holding against an implausible `Retry-After`, and a non-numeric header falling back to the schedule.

Request construction: the prompt carries every expected `doc_id` and the schema map, the allowed-keyword list is sorted so the prompt is stable across runs, and `json_mode` reaches the client as `response_format` only when it is enabled.

Accounting: one case runs two batches through one counter object, one succeeding and one exhausting its retries, to verify neither result bleeds into the other's metrics.

### Reconciliation Outcomes

Thirteen cases cover the five reachable reconciliation paths and the `review_reason` priority ladder: eight on the paths themselves, including two that check urgency is not special-cased in either direction, and five on which reason wins when several apply at once. The behaviours they pin down:

- Keyword and LLM agree: `conflict` is false, `review_reason` is null, record goes to `miscellaneous.json` only.
- No keyword hit, LLM valid with non-urgent label: record goes to `miscellaneous.json` only.
- No keyword hit, LLM labels the document urgent: record goes to `urgent.json` only with no review flag.
- Keyword and LLM disagree with the keyword signal strong enough to challenge the model: conflict flag is raised, LLM wins the route, confidence is penalised, `review_reason` is `"conflict"`, record goes to `human_review.json`.
- LLM API exhausted with a keyword signal present: keyword fallback provides the route, `classification.llm` is null, `error_reason` and `review_reason` both equal `"llm_api_error"`, record goes to `human_review.json`.
- Envelope failure with no keyword signal: double fallback to `route == "general"`, `final_confidence == 0.0`, `review_reason == "llm_envelope_error"`.

### Per-Document Quality Failures

Three cases cover the document-level failure modes that do not take down the whole batch. One omits an expected `doc_id` from the response: the missing document is marked with `error_reason == "llm_response_missing"`, `review_reason == "llm_missing_docs"`, and the `llm_missing_docs` counter advances by one. One returns an unknown `doc_id` alongside the real ones: the hallucinated result is dropped without any counter moving. One returns a well-formed result whose `extracted_fields` violates the per-category schema: the record retains its classification context, `extracted_fields` is set to `null`, and both `error_reason` and `review_reason` become `"llm_schema_mismatch"`.

### Output Partitioning

The partition rules are asserted throughout the pipeline cases rather than in one place: every run helper reloads all four written files from disk, and `assert_record_only_in` names the files a given record must and must not appear in. `miscellaneous.json` is the disjoint remainder, `urgent.json` and `human_review.json` may overlap, and between them the three cover every record. One further case reads the four files back off the configured output directory and checks their contents there.

### Whole-Run Behaviour

Four cases exercise the pipeline end to end with the LLM boundary stubbed. One runs a mixed set of sample documents through every reconciliation outcome at once. One checks that an `urgent` label reaches `urgent.json` as a route rather than only as a review flag. One feeds two files with identical content: because `doc_id` is a hash of the content, both records carry the same id and both read back the single model result — two records, two source paths, no `error_reason`, `input_file_ids` keeping the duplicate, and a warning as the only signal to the operator. The last runs the shipped `sample_docs/` against a canned response set with an injected clock, then asserts the full record schema, the partition counts, the ordering, and every metric — the regression net that catches an unintended change anywhere in the contract.

## Edge Case Handling

Several behaviours are worth flagging because naive implementations get them wrong:

**Counting hallucinated doc_ids.** Unknown returned IDs are dropped without metric increment; missing expected IDs are counted. If both sides counted, a model that hallucinated an ID to replace a missing one would be credited twice.

**Whole-batch vs per-document failure.** The pydantic envelope deliberately does not enforce the `keyword` enum. An off-category value from the model becomes a per-document schema mismatch, which keeps the rest of the batch usable; enforcing the enum at the envelope would collapse the whole batch over a single bad item.

**Batch failure visibility.** A batch-level LLM failure bypasses the normal confidence threshold by mirroring `error_reason` into `review_reason` at the highest priority. Without this, a confident keyword fallback could produce a high `final_confidence` on a document the model never actually saw, letting it land in `miscellaneous.json` unreviewed.

**Write-failure deduplication.** A record that appears in both `urgent.json` and `human_review.json` is one delivery. When both files fail to write, the logic tracks first-failure per `(doc_id, source_path)` so only one `file_errors` increment occurs. Without this, overlap between the two views would inflate the counter and break the consistency invariant any consumer might rely on.

**Whitespace-only content is not empty.** The gate rejects the empty string; it does not reject `"   \n"`. Whether this matters depends on how aggressively the downstream wants to pre-filter; the pipeline chooses the permissive definition so the model sees what the filesystem contains.
